#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Yeogigangwon data-pipeline: 수집 → 정제/표준화 → 시간 피처 → 조인 → 학습셋 CSV!

사용 예:
  python pipeline.py odcloud
  python pipeline.py kto --area 32 --rows 200
  python pipeline.py asos --start 2024-06-01 --end 2024-08-31 --stn 105 90
  python pipeline.py join --area 32

개발 메모:
- serviceKey는 코드 내부에서 URL 인코딩 처리하지 않아도 되도록, requests의 params에 그대로 전달함
  (requests가 안전하게 처리). 단, 키에 '+' 등 특수문자가 있으면 인코딩이 필요할 수 있으니
  .env의 값을 있는 그대로 넣고, 문제가 생기면 encode 시도할 것.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from ruamel.yaml import YAML
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

from urllib.parse import unquote

def normalize_service_key(k: str) -> str:
    """%252F 처럼 이중/다중 인코딩된 키를 원복(완전 디코딩)한다."""
    if not k:
        return k
    prev = None
    cur = k
    # '%25'가 사라질 때까지 반복 디코딩
    while prev != cur:
        prev = cur
        cur = unquote(cur)
    return cur

# ---------------------------
# 설정 및 경로 유틸
# ---------------------------
def load_settings() -> dict:
    cfg_path = Path("config/settings.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError("config/settings.yaml 이 없습니다!")
    yaml = YAML(typ="safe")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.load(f)

def ensure_dirs(settings: dict) -> None:
    for key in ("output_dir", "raw_dir", "interim_dir", "processed_dir"):
        Path(settings["paths"][key]).mkdir(parents=True, exist_ok=True)

# ---------------------------
# HTTP 클라이언트(재시도 포함)
# ---------------------------
class HttpError(Exception):
    pass

def _timeout_tuple(settings: dict):
    return (settings["http"]["connect_timeout_sec"], settings["http"]["read_timeout_sec"])

@retry(
    reraise=True,
    retry=retry_if_exception_type(HttpError),
    wait=wait_exponential(multiplier=1, min=0.5, max=20),
    stop=stop_after_attempt(5),
)
def http_get_json(url: str, params: dict, settings: dict) -> dict:
    """JSON 응답 전제. XML/HTML이 오면 예외! 429/5xx는 재시도!"""
    try:
        r = requests.get(url, params=params, timeout=_timeout_tuple(settings))
    except requests.RequestException as e:
        raise HttpError(f"요청 실패: {e}") from e

    if r.status_code in (429, 500, 502, 503, 504):
        raise HttpError(f"서버/쿼터 오류로 재시도: {r.status_code} {r.text[:200]}")

    ct = r.headers.get("Content-Type", "").lower()
    if "json" not in ct:
        # 공공데이터는 에러 시 XML로 주는 경우가 잦음 → 내용 로그
        raise HttpError(f"JSON 아님(Content-Type={ct}) body={r.text[:200]}")

    try:
        return r.json()
    except json.JSONDecodeError as e:
        raise HttpError(f"JSON 파싱 실패: {e} body={r.text[:200]}") from e

# ---------------------------
# ODCLOUD (해수욕장 메타)
# ---------------------------
def collect_odcloud(settings: dict, service_key: str) -> pd.DataFrame:
    base = settings["sources"]["odcloud"]["base"]
    path = settings["sources"]["odcloud"]["path"]
    per_page = settings["sources"]["odcloud"]["per_page"]
    raw_dir = Path(settings["paths"]["raw_dir"]) / "odcloud"
    interim_dir = Path(settings["paths"]["interim_dir"]) / "odcloud"
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    page = 1
    all_rows: List[dict] = []
    while True:
        url = base + path
        params = {
            "page": page,
            "perPage": per_page,
            "returnType": "JSON",
            "serviceKey": service_key,
        }
        data = http_get_json(url, params, settings)
        rows = data.get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        snap = raw_dir / f"odcloud_page{page}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        snap.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        if len(rows) < per_page:
            break
        page += 1

    if not all_rows:
        print("[ODCLOUD] 데이터 없음!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # -------- 컬럼 매핑 (이번 응답 키에 맞춰 강화) --------
    def pick(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    name_col   = pick(["해수욕장명", "해수욕장명칭", "해수욕장", "beachName", "name"])
    # 면적/길이/너비 후보에 '백사장면적(m2)' 포함!
    area_col   = pick(["백사장면적(㎡)", "백사장면적(m2)", "백사장면적", "면적", "area", "sandAreaM2"])
    length_col = pick(["길이(m)", "길이", "length_m", "length"])
    width_col  = pick(["너비(m)", "너비", "width_m", "width"])
    lat_col    = pick(["위도", "lat", "LAT"])
    lon_col    = pick(["경도", "lon", "LON"])

    std = pd.DataFrame()

    if name_col:
        std["beach_name_src"] = df[name_col].astype(str)

    # 숫자 변환 보조: 콤마/공백 제거 후 숫자화
    def to_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")

    # 1) 면적 바로 매핑
    if area_col:
        std["sand_area_m2"] = to_num(df[area_col])

    # 2) 면적이 비거나 전부 NaN이면 길이×너비로 보간
    need_fill = ("sand_area_m2" not in std.columns) or std["sand_area_m2"].isna().all()
    if need_fill and length_col and width_col:
        L = to_num(df[length_col])
        W = to_num(df[width_col])
        std["sand_area_m2"] = L * W

    # 좌표(있으면 매핑)
    if lat_col:
        std["lat"] = to_num(df[lat_col])
    if lon_col:
        std["lon"] = to_num(df[lon_col])

    # 표준명/ID 생성
    def normalize_name(x: str) -> str:
        x = re.sub(r"\s+", "", x)
        x = x.replace("해수욕장", "")
        return x

    if "beach_name_src" in std:
        std["beach_name_std"] = std["beach_name_src"].map(normalize_name)
        std["beach_id"] = std["beach_name_std"].str.lower()

    # 중복 제거
    if "beach_id" in std:
        std = std.drop_duplicates(subset=["beach_id"]).reset_index(drop=True)

    # 저장
    interim_path = interim_dir / f"beach_meta_{datetime.now().strftime('%Y%m%d')}.csv"
    std.to_csv(interim_path, index=False, encoding="utf-8-sig")
    print(f"[ODCLOUD] 표준화 저장: {interim_path}")
    return std


# ---------------------------
# 한국관광공사 (집중률 예측)
# ---------------------------
def collect_kto(settings: dict, service_key: str, area_cd: str, num_rows: int,
                sigungu_cd: Optional[str], tAtsNm: Optional[str]) -> pd.DataFrame:
    """
    한국관광공사 '관광지 집중률 방문자 추이 예측 정보' 수집/표준화.

    - signguCd(시군구코드) 필수: 인자로 없으면 settings.sources.kto.signgu_codes[area_cd] 순회
    - tAtsNm(관광지명) 선택
    - 응답이 JSON / JSON-string / XML(에러 포함) 모두 안전 파싱
    - SSL시 http↔https 폴백, 서비스키 다중 인코딩 정규화
    - 스냅샷 저장 및 표준화 출력
    """
    from datetime import datetime
    import glob, xmltodict

    base = settings["sources"]["kto"]["base"]     # 매뉴얼 기준 http 권장
    path = settings["sources"]["kto"]["path"]
    mobile_os = settings["sources"]["kto"]["mobile_os"]
    mobile_app = settings["sources"]["kto"]["mobile_app"]

    raw_dir = Path(settings["paths"]["raw_dir"]) / "kto"
    interim_dir = Path(settings["paths"]["interim_dir"]) / "kto"
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    # ---- serviceKey 정규화 (이중 인코딩 방지). 인코딩 키를 그대로 쓰려면 USE_ENCODED_KTO_KEY=true ----
    use_encoded = os.getenv("USE_ENCODED_KTO_KEY", "").lower() == "true"
    skey = service_key if use_encoded else normalize_service_key(service_key)

    # ---- signguCd 결정 (필수) ----
    if sigungu_cd:
        signgu_list = [str(sigungu_cd)]
    else:
        signgu_map = settings["sources"]["kto"].get("signgu_codes", {})
        signgu_list = [str(x) for x in signgu_map.get(str(area_cd), [])]
        if not signgu_list:
            raise ValueError(
                "signguCd(시군구 코드)가 필요합니다. "
                "--sigungu <코드>를 주거나, config/settings.yaml의 "
                "sources.kto.signgu_codes[area_cd]에 목록을 채워주세요."
            )

    # ---- 요청/파싱 보조 ----
    def try_request(url: str, params: dict) -> dict:
        """http_get_json 사용 + SSL/핸드셰이크 시 스킴 폴백."""
        try:
            return http_get_json(url, params, settings)
        except HttpError as e:
            msg = str(e).lower()
            if "ssl" in msg or "handshake" in msg:
                alt = url.replace("https://", "http://") if url.startswith("https://") else url.replace("http://", "https://")
                return http_get_json(alt, params, settings)
            raise

    def to_dict(data: Any) -> dict:
        """응답을 dict로 정규화: dict → 그대로, str → JSON 시도 → XML 시도."""
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            # JSON 문자열 시도
            try:
                return json.loads(data)
            except Exception:
                pass
            # XML 문자열 시도
            try:
                return xmltodict.parse(data)
            except Exception:
                # 파싱 실패 시 원문을 담아 반환(에러 메시지로 노출)
                return {"raw": data}
        # 그 외 타입 방어
        return {"raw": str(data)}

    def raise_if_openapi_error(doc: dict) -> None:
        """OpenAPI_ServiceResponse(XML 에러 포맷)이면 명확한 예외 발생."""
        svc = doc.get("OpenAPI_ServiceResponse")
        if not svc:
            return
        hdr = svc.get("cmmMsgHeader", {})
        err = hdr.get("returnAuthMsg") or hdr.get("errMsg") or "UNKNOWN_ERROR"
        code = hdr.get("returnReasonCode")
        raise HttpError(f"[KTO] OpenAPI 오류: {err} (code={code})")

    def extract_items(doc: dict) -> List[dict]:
        """표준 JSON 구조에서 items.item을 안전하게 추출."""
        resp = doc.get("response") or {}
        body = resp.get("body") or {}
        items = body.get("items") or {}
        it = items.get("item")
        if it is None:
            return []
        if isinstance(it, list):
            return it
        return [it]  # 단일 객체일 때

    def fetch_for_signgu(signgu: str, extra_params: dict) -> List[dict]:
        page_no = 1
        items_all: List[dict] = []
        while True:
            url = base + path
            params = {
                "serviceKey": skey,
                "pageNo": page_no,
                "numOfRows": num_rows,
                "MobileOS": mobile_os,
                "MobileApp": mobile_app,
                "areaCd": area_cd,
                "signguCd": signgu,   # ← 필수 파라미터
                "_type": "json",
            }
            if tAtsNm:
                params["tAtsNm"] = tAtsNm
            params.update(extra_params)

            raw = try_request(url, params)
            doc = to_dict(raw)

            # 스냅샷 저장(원문 그대로)
            snap = raw_dir / f"kto_{signgu}_page{page_no}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            try:
                snap.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            except Exception:
                # dict 직렬화 실패 시 문자열로 저장
                snap.write_text(str(doc), encoding="utf-8")

            # XML 에러 포맷이면 명확히 실패시킴
            raise_if_openapi_error(doc)

            items = extract_items(doc)
            if not items:
                break
            items_all.extend(items)
            if len(items) < num_rows:
                break
            page_no += 1
        return items_all

    # ---- 1) 기본 수집 ----
    items: List[dict] = []
    for sgg in signgu_list:
        items += fetch_for_signgu(sgg, extra_params={})

    # ---- 2) 비어 있으면 baseYmd=오늘로 재시도(운영상 날짜 요구 케이스 대비) ----
    if not items:
        today = datetime.now().strftime("%Y%m%d")
        for sgg in signgu_list:
            items += fetch_for_signgu(sgg, extra_params={"baseYmd": today})

    if not items:
        print("[KTO] 데이터 없음!")
        return pd.DataFrame()

    df = pd.DataFrame(items)

    # -------- 표준화 --------
    def pick(cols: List[str]) -> Optional[str]:
        for c in cols:
            if c in df.columns:
                return c
        return None

    name_col = pick(["tAtsNm", "관광지명", "name", "title"])
    ts_col   = pick(["baseYmd", "baseDe", "ymd", "date", "obsrDe"])
    pred_col = pick(["cnctrRate", "cnctrRatePred", "visitorRate", "predicted", "value"])

    std = pd.DataFrame()
    if name_col:
        std["place_name_src"] = df[name_col].astype(str)
    if ts_col:
        std["date"] = pd.to_datetime(df[ts_col].astype(str), errors="coerce").dt.date
    if pred_col:
        std["visitor_rate_pred"] = pd.to_numeric(df[pred_col], errors="coerce")

    def normalize_beach_name(x: str) -> str:
        x0 = re.sub(r"\s+", "", x)
        return x0.replace("해수욕장", "")

    if "place_name_src" in std:
        std["beach_name_std"] = std["place_name_src"].map(normalize_beach_name)
        std["beach_id"] = std["beach_name_std"].str.lower()
    if "date" in std.columns:
        std = std.dropna(subset=["date"]).copy()

    out_path = interim_dir / f"kto_{area_cd}_{datetime.now().strftime('%Y%m%d')}.csv"
    std.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[KTO] 표준화 저장: {out_path}")
    return std



# ---------------------------
# 기상청 ASOS 일자료
# ---------------------------
def daterange_days(d0: date, d1: date) -> List[date]:
    cur = d0
    out = []
    while cur <= d1:
        out.append(cur)
        cur += timedelta(days=1)
    return out

def collect_kma_asos_daily(settings: dict, service_key: str,
                           stn_ids: List[str], start: date, end: date) -> pd.DataFrame:
    """
    기상청 ASOS 일자료 조회 서비스 수집/표준화.
    - 입력: 다수 지점(stn_ids), 시작/종료일(YYYY-MM-DD)
    - 출력: 지점별 CSV 저장 + 모든 지점 concat한 DataFrame 반환
    - 기능: 서비스키 다중 인코딩 정규화, http↔https 폴백, XML/JSON 안전파싱
    - 저장 파일명: output/interim/kma/asos_daily_{stn}_{start}_{end}.csv
    - 표준 컬럼: ['asos_stn_id','date','tavg','tmin','tmax','rain_sum','wind_avg','humid_avg']
    """
    from datetime import datetime
    import xmltodict

    base = settings["sources"]["kma"]["base"]          # 예: http://apis.data.go.kr/1360000/AsosDalyInfoService
    path = settings["sources"]["kma"]["path"]          # 예: /getWthrDataList
    raw_dir = Path(settings["paths"]["raw_dir"]) / "kma"
    interim_dir = Path(settings["paths"]["interim_dir"]) / "kma"
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    # ---- serviceKey 정규화 (인코딩키 그대로 쓰려면 USE_ENCODED_KMA_KEY=true) ----
    use_encoded = os.getenv("USE_ENCODED_KMA_KEY", "").lower() == "true"
    skey = service_key if use_encoded else normalize_service_key(service_key)

    def try_request(url: str, params: dict) -> Any:
        """requests 기반 http_get_json 사용 + SSL/핸드셰이크 시 스킴 폴백."""
        try:
            return http_get_json(url, params, settings)
        except HttpError as e:
            msg = str(e).lower()
            if "ssl" in msg or "handshake" in msg:
                alt = url.replace("https://", "http://") if url.startswith("https://") else url.replace("http://", "https://")
                return http_get_json(alt, params, settings)
            raise

    def to_dict(data: Any) -> dict:
        """응답을 dict로 정규화: dict → 그대로, str → JSON 시도 → XML 시도."""
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            try:
                return json.loads(data)
            except Exception:
                pass
            try:
                return xmltodict.parse(data)
            except Exception:
                return {"raw": data}
        return {"raw": str(data)}

    def raise_if_openapi_error(doc: dict) -> None:
        """OpenAPI_ServiceResponse(XML 에러 포맷)이면 명확한 예외 발생."""
        svc = doc.get("OpenAPI_ServiceResponse")
        if not svc:
            return
        hdr = svc.get("cmmMsgHeader", {})
        err = hdr.get("returnAuthMsg") or hdr.get("errMsg") or "UNKNOWN_ERROR"
        code = hdr.get("returnReasonCode")
        raise HttpError(f"[KMA] OpenAPI 오류: {err} (code={code})")

    def extract_items(doc: dict) -> List[dict]:
        """표준 JSON 구조에서 items.item을 안전 추출."""
        resp = doc.get("response") or {}
        body = resp.get("body") or {}
        items = body.get("items") or {}
        it = items.get("item")
        if it is None:
            return []
        if isinstance(it, list):
            return it
        return [it]  # 단일 객체

    def page_fetch(stn: str) -> pd.DataFrame:
        page_no = 1
        rows_all: List[dict] = []
        num_rows = settings.get("kma", {}).get("num_rows", 500)  # 없으면 기본 500
        start_str = start.strftime("%Y%m%d")
        end_str = end.strftime("%Y%m%d")

        while True:
            url = base + path
            params = {
                "serviceKey": skey,
                "pageNo": page_no,
                "numOfRows": num_rows,
                "dataType": "JSON",
                "dataCd": "ASOS",
                "dateCd": "DAY",
                "startDt": start_str,
                "endDt": end_str,
                "stnIds": str(stn),
            }
            raw = try_request(url, params)
            doc = to_dict(raw)

            # 스냅샷 저장
            snap = raw_dir / f"asos_daily_{stn}_page{page_no}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            try:
                snap.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            except Exception:
                snap.write_text(str(doc), encoding="utf-8")

            raise_if_openapi_error(doc)
            items = extract_items(doc)
            if not items:
                break
            rows_all.extend(items)
            if len(items) < num_rows:
                break
            page_no += 1

        if not rows_all:
            return pd.DataFrame()

        # ---- 표준화 매핑 ----
        df = pd.DataFrame(rows_all)

        def pick(cols: List[str]) -> Optional[str]:
            for c in cols:
                if c in df.columns:
                    return c
            return None

        # KMA 컬럼 표준명 후보
        tm_col     = pick(["tm", "date"])              # 날짜
        avgTa_col  = pick(["avgTa", "avgta", "tavg"])  # 평균기온
        minTa_col  = pick(["minTa", "tmn", "tmin"])
        maxTa_col  = pick(["maxTa", "tmx", "tmax"])
        sumRn_col  = pick(["sumRn", "rn_day", "rain_sum"])
        avgWs_col  = pick(["avgWs", "ws", "wind_avg"])
        avgRhm_col = pick(["avgRhm", "rhm", "humid_avg"])

        out = pd.DataFrame()
        if tm_col:
            out["date"] = pd.to_datetime(df[tm_col].astype(str), errors="coerce").dt.date
        if avgTa_col:
            out["tavg"] = pd.to_numeric(df[avgTa_col], errors="coerce")
        if minTa_col:
            out["tmin"] = pd.to_numeric(df[minTa_col], errors="coerce")
        if maxTa_col:
            out["tmax"] = pd.to_numeric(df[maxTa_col], errors="coerce")
        if sumRn_col:
            out["rain_sum"] = pd.to_numeric(df[sumRn_col], errors="coerce")
        if avgWs_col:
            out["wind_avg"] = pd.to_numeric(df[avgWs_col], errors="coerce")
        if avgRhm_col:
            out["humid_avg"] = pd.to_numeric(df[avgRhm_col], errors="coerce")

        # 지점 코드 추가
        out["asos_stn_id"] = str(stn)

        # 저장
        out_path = interim_dir / f"asos_daily_{stn}_{start_str}_{end_str}.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[KMA] 표준화 저장: {out_path}")
        return out

    # ---- 지점별 수집 + 결합 ----
    all_df: List[pd.DataFrame] = []
    for stn in stn_ids:
        part = page_fetch(str(stn))
        if not part.empty:
            all_df.append(part)

    if not all_df:
        print("[KMA] 수집 결과가 비었습니다.")
        return pd.DataFrame()

    return pd.concat(all_df, ignore_index=True)

# ---------------------------
# 시간 파생 피처
# ---------------------------
def add_time_features(df: pd.DataFrame, date_col: str = "date", hour_col: Optional[str] = None,
                      peak_hours: Optional[List[int]] = None) -> pd.DataFrame:
    out = df.copy()
    if date_col in out:
        dts = pd.to_datetime(out[date_col], errors="coerce")
        out["year"] = dts.dt.year
        out["month"] = dts.dt.month
        out["day"] = dts.dt.day
        out["dow"] = dts.dt.dayofweek  # 0=월
        out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)
        # season: 12~2 겨울/ 3~5 봄/ 6~8 여름/ 9~11 가을
        def season_of(m):
            if m in (12, 1, 2): return "winter"
            if m in (3, 4, 5): return "spring"
            if m in (6, 7, 8): return "summer"
            return "autumn"
        out["season"] = out["month"].map(season_of)
    if hour_col and hour_col in out:
        out["hour_bucket"] = out[hour_col].apply(lambda h: "peak" if (peak_hours and int(h) in peak_hours) else ("am" if int(h) < 12 else "pm"))
    return out

# ---------------------------
# 간단 매핑(데모용)
# 실제 운영에서는 mapping_beach_asos.yaml, mapping_beach_alias.yaml 사용 권장
# ---------------------------
DEFAULT_BEACH_TO_ASOS = {
    # beach_id(소문자/공백제거) : ASOS 지점ID
    # 예시: "속초": "90", "강릉": "105"
}

def attach_asos_to_beach(beach_df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    df = beach_df.copy()
    df["asos_stn_id"] = df.get("beach_id", pd.Series([None]*len(df))).map(mapping)
    return df

# ---------------------------
# 조인: beach_id + date 기준
# ---------------------------
def build_training_table(settings: dict,
                         od_map: pd.DataFrame,      # ODCLOUD 메타(해변별 고정 정보)
                         kto_df: pd.DataFrame,      # KTO 집중률(관광지/날짜)
                         asos_df: pd.DataFrame,     # ASOS 일자료(지점/날짜)
                         area_cd: str) -> pd.DataFrame:
    """
    통합 학습 테이블 생성:
      - 필수: beach_id, sand_area_m2, date
      - 선택: lat, lon, asos_stn_id (없으면 NaN으로 채워 진행)
      - ASOS 매핑:
          * od_map에 asos_stn_id가 있으면 (date, asos_stn_id)로 조인
          * 없으면 ASOS를 날짜별 평균치로 뽑아 전체 해변에 동일 적용
      - KTO가 없어도(빈 DF) 피처셋만 생성
    저장: output/processed/train_dataset_area{area_cd}_YYYYMMDD.csv
    """
    from datetime import datetime

    # ---- 0) ODCLOUD 메타 선택 컬럼 안전 선택 ----
    base_cols = ["beach_id", "beach_name_std", "sand_area_m2"]
    opt_cols  = ["lat", "lon", "asos_stn_id"]

    for c in base_cols:
        if c not in od_map.columns:
            raise ValueError(f"[JOIN] ODCLOUD 메타에 '{c}' 컬럼이 없습니다. 수집/표준화를 확인해주세요.")

    present_cols = base_cols + [c for c in opt_cols if c in od_map.columns]
    od_sel = od_map[present_cols].copy()
    # 누락 선택 컬럼은 NaN으로 추가
    for c in opt_cols:
        if c not in od_sel.columns:
            od_sel[c] = pd.NA

    # ---- 1) 날짜·해변 베이스 인덱스 만들기 ----
    # 1) KTO가 있으면 그 날짜·해변을 우선 사용
    have_kto = not kto_df.empty and {"beach_id", "date"}.issubset(kto_df.columns)
    if have_kto:
        kto_df = kto_df.copy()
        kto_df["date"] = pd.to_datetime(kto_df["date"], errors="coerce").dt.date
        base_idx = kto_df[["beach_id", "date"]].dropna().drop_duplicates()
    else:
        # 2) KTO가 없으면 ASOS 날짜범위 × 모든 해변으로 기본 인덱스 생성
        if asos_df.empty or "date" not in asos_df.columns:
            raise ValueError("[JOIN] KTO도 없고 ASOS도 없습니다. 최소 한 소스의 날짜가 필요합니다.")
        dt = pd.to_datetime(asos_df["date"], errors="coerce").dt.date
        days = pd.Series(sorted(dt.dropna().unique()), name="date")
        beaches = od_sel["beach_id"].dropna().unique()
        base_idx = pd.MultiIndex.from_product([beaches, days], names=["beach_id", "date"]).to_frame(index=False)

    # ---- 2) ASOS 준비: 매핑 유무에 따라 전략 분기 ----
    asos_df = asos_df.copy()
    asos_df["date"] = pd.to_datetime(asos_df["date"], errors="coerce").dt.date

    # 수치 컬럼만 평균 계산 대상으로
    asos_num = asos_df.select_dtypes(include="number").columns.tolist()
    # date, asos_stn_id는 키 컬럼으로 유지
    if "asos_stn_id" in asos_df.columns:
        key_cols = ["date", "asos_stn_id"]
    else:
        key_cols = ["date"]

    # 그룹 집계 (지점별 평균 혹은 날짜 평균)
    if "asos_stn_id" in asos_df.columns:
        wthr = asos_df.groupby(key_cols, as_index=False)[asos_num].mean(numeric_only=True)
    else:
        wthr = asos_df.groupby("date", as_index=False)[asos_num].mean(numeric_only=True)

    # ---- 3) 베이스 인덱스 ← 해변 메타 조인 ----
    df = base_idx.merge(od_sel, on="beach_id", how="left")

    # ---- 4) 베이스 인덱스 ← KTO 조인(있으면) ----
    if have_kto and "visitor_rate_pred" in kto_df.columns:
        df = df.merge(
            kto_df[["beach_id", "date", "visitor_rate_pred"]],
            on=["beach_id", "date"], how="left"
        )
    else:
        df["visitor_rate_pred"] = pd.NA  # KTO 없으면 타겟 결측으로 둔다

    # ---- 5) 베이스 인덱스 ← ASOS 조인 ----
    if "asos_stn_id" in df.columns and df["asos_stn_id"].notna().any() and "asos_stn_id" in wthr.columns:
        # 해변별 관측소 매핑이 있을 때: (date, asos_stn_id)로 정밀 조인
        df = df.merge(wthr, on=["date", "asos_stn_id"], how="left", suffixes=("", "_w"))
    else:
        # 매핑이 없을 때: 날짜 평균 날씨를 전 해변 공통 적용
        df = df.merge(wthr, on=["date"], how="left", suffixes=("", "_w"))

    # ---- 6) 시간 파생 피처 ----
    dt = pd.to_datetime(df["date"], errors="coerce")
    df["year"]  = dt.dt.year
    df["month"] = dt.dt.month
    df["dow"]   = dt.dt.weekday  # 월=0, 일=6
    df["is_weekend"] = df["dow"].isin([5, 6]).astype("Int64")

    # 시즌(임시 규칙): 6~8=성수기, 9~10=가을, 11~2=비수기, 3~5=봄
    def season_of(m):
        if pd.isna(m):
            return pd.NA
        m = int(m)
        if 6 <= m <= 8:
            return "peak"
        if 9 <= m <= 10:
            return "autumn"
        if m in (11, 12, 1, 2):
            return "off"
        return "spring"
    df["season"] = df["month"].map(season_of)

    # ---- 7) 저장 ----
    out_dir = Path(settings["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"train_dataset_area{area_cd}_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[JOIN] 최종 저장: {out_path}  (rows={len(df)})")

    return df


# ---------------------------
# CLI
# ---------------------------
def main():
    # .env 로드
    if Path(".env").exists():
        load_dotenv(".env")
        print("[ENV] .env 로드 완료!")
    else:
        print("[ENV] .env 없음(.env.example을 복사해 생성하세요)!")

    settings = load_settings()
    ensure_dirs(settings)

    parser = argparse.ArgumentParser(description="Yeogigangwon data-pipeline CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # odcloud
    sub.add_parser("odcloud", help="해양수산부 해변 메타 수집/표준화")

    # kto
    p_kto = sub.add_parser("kto", help="관광공사 집중률 예측 수집/표준화")
    p_kto.add_argument("--area", default=os.getenv("DEFAULT_AREA_CD", "32"), help="areaCd (기본: 32=강원)")
    p_kto.add_argument("--rows", type=int, default=int(os.getenv("DEFAULT_NUM_ROWS", "200")), help="numOfRows")
    p_kto.add_argument("--sigungu", default=None, help="sigunguCd (선택)")
    p_kto.add_argument("--name", default=None, help="tAtsNm (선택, 관광지명)")

    # asos
    p_asos = sub.add_parser("asos", help="기상청 ASOS 일자료 수집/표준화")
    p_asos.add_argument("--start", required=True, help="YYYY-MM-DD")
    p_asos.add_argument("--end", required=True, help="YYYY-MM-DD")
    p_asos.add_argument("--stn", nargs="*", default=None, help="ASOS 지점 ID 목록(공백 구분)")

    # join
    p_join = sub.add_parser("join", help="표준화된 소스들을 조인해 최종 학습셋 생성")
    p_join.add_argument("--area", default=os.getenv("DEFAULT_AREA_CD", "32"), help="areaCd (파일명 태깅용)")

    args = parser.parse_args()

    # 서비스키 확보
    od_key = os.getenv("SECRETS_ODCLOUD_KEY", "")
    kto_key = os.getenv("SECRETS_KTO_KEY", "")
    kma_key = os.getenv("SECRETS_KMA_KEY", "")

    if args.cmd == "odcloud":
        if not od_key:
            print("SECRETS_ODCLOUD_KEY 가 .env에 없습니다!", file=sys.stderr)
            sys.exit(1)
        collect_odcloud(settings, od_key)

    elif args.cmd == "kto":
        if not kto_key:
            print("SECRETS_KTO_KEY 가 .env에 없습니다!", file=sys.stderr)
            sys.exit(1)
        collect_kto(settings, kto_key, area_cd=args.area, num_rows=args.rows,
                    sigungu_cd=args.sigungu, tAtsNm=args.name)

    elif args.cmd == "asos":
        if not kma_key:
            print("SECRETS_KMA_KEY 가 .env에 없습니다!", file=sys.stderr)
            sys.exit(1)
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        stn_list = args.stn or []
        if not stn_list:
            print("[KMA] --stn 에 ASOS 지점 ID를 지정하세요! 예: --stn 90 105", file=sys.stderr)
            sys.exit(1)
        collect_kma_asos_daily(settings, kma_key, stn_list, start, end)

    elif args.cmd == "join":
        # 중간 산출물 읽기
        # ODCLOUD (해수욕장 표준 메타)
        od_dir = Path(settings["paths"]["interim_dir"]) / "odcloud"
        od_files = sorted(od_dir.glob("beach_meta_*.csv"))
        if not od_files:
            print("[JOIN] odcloud 표준화 파일이 없습니다! 먼저 python pipeline.py odcloud 실행!", file=sys.stderr)
            sys.exit(1)
        od_df = pd.read_csv(od_files[-1], encoding="utf-8")

        # ⬇️ [추가] od_df (해수욕장 목록)에서 강원특별자치도 데이터만 남기도록 필터링 ⬇️
        print("[JOIN] 해수욕장 목록을 '강원특별자치도'로 필터링합니다.")
        od_df = od_df[od_df["sido_nm"] == "강원특별자치도"].copy()

        # KTO
        kto_dir = Path(settings["paths"]["interim_dir"]) / "kto"
        kto_files = sorted(kto_dir.glob("kto_*/*.csv"))  # 서브폴더 없음 대비 아래 줄도 함께 검사
        if not kto_files:
            kto_files = sorted(kto_dir.glob("kto_*.csv"))
        if not kto_files:
            print("[JOIN] kto 표준화 파일이 없습니다! 먼저 python pipeline.py kto 실행!", file=sys.stderr)
            sys.exit(1)
        kto_df = pd.read_csv(kto_files[-1], encoding="utf-8")
        # 날짜 컬럼 보정
        if "date" in kto_df.columns:
            kto_df["date"] = pd.to_datetime(kto_df["date"], errors="coerce").dt.date

        # KMA
        kma_dir = Path(settings["paths"]["interim_dir"]) / "kma"
        kma_files = sorted(kma_dir.glob("asos_*.csv"))
        if kma_files:
            asos_df = pd.read_csv(kma_files[-1], encoding="utf-8")
            if "date" in asos_df.columns:
                asos_df["date"] = pd.to_datetime(asos_df["date"], errors="coerce").dt.date
        else:
            asos_df = pd.DataFrame()

        build_training_table(settings, od_df, kto_df, asos_df, area_cd=args.area)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
