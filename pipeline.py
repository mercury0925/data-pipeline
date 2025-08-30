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
        # odcloud는 {currentCount, data: [...]} 형태
        rows = data.get("data", [])
        if not rows:
            break
        all_rows.extend(rows)
        # 스냅샷 저장
        snap = raw_dir / f"odcloud_page{page}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        snap.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        if len(rows) < per_page:
            break
        page += 1

    if not all_rows:
        print("[ODCLOUD] 데이터 없음!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # 기본 표준화: 컬럼 추정(문서에 따라 컬럼명이 다를 수 있음 → 존재하면 매핑)
    # 후보 컬럼: 해수욕장명, 주소, 면적(㎡), 위도/경도 등
    def pick(colnames: List[str]) -> Optional[str]:
        for c in colnames:
            if c in df.columns:
                return c
        return None

    name_col = pick(["해수욕장명", "해수욕장명칭", "해수욕장", "beachName", "name"])
    area_col = pick(["백사장면적(㎡)", "백사장면적", "area", "sandAreaM2"])
    lat_col  = pick(["위도", "lat", "LAT"])
    lon_col  = pick(["경도", "lon", "LON"])

    std = pd.DataFrame()
    if name_col is not None:
        std["beach_name_src"] = df[name_col].astype(str)
    if area_col is not None:
        std["sand_area_m2"] = pd.to_numeric(df[area_col], errors="coerce")
    if lat_col is not None:
        std["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    if lon_col is not None:
        std["lon"] = pd.to_numeric(df[lon_col], errors="coerce")

    # 내부 표준명/ID 생성(간단 정규화)
    def normalize_name(x: str) -> str:
        x = re.sub(r"\s+", "", x)
        x = x.replace("해수욕장", "")
        return x

    if "beach_name_src" in std:
        std["beach_name_std"] = std["beach_name_src"].map(normalize_name)
        std["beach_id"] = std["beach_name_std"].str.lower()

    # 중복 제거
    if "beach_id" in std:
        std = std.drop_duplicates(subset=["beach_id"])

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
    base = settings["sources"]["kto"]["base"]
    path = settings["sources"]["kto"]["path"]
    mobile_os = settings["sources"]["kto"]["mobile_os"]
    mobile_app = settings["sources"]["kto"]["mobile_app"]

    raw_dir = Path(settings["paths"]["raw_dir"]) / "kto"
    interim_dir = Path(settings["paths"]["interim_dir"]) / "kto"
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    page_no = 1
    all_items: List[dict] = []
    while True:
        url = base + path
        params = {
            "serviceKey": service_key,
            "pageNo": page_no,
            "numOfRows": num_rows,
            "MobileOS": mobile_os,
            "MobileApp": mobile_app,
            "areaCd": area_cd,
            "_type": "json",
        }
        if sigungu_cd:
            params["sigunguCd"] = sigungu_cd
        if tAtsNm:
            params["tAtsNm"] = tAtsNm

        data = http_get_json(url, params, settings)

        # KTO 표준 응답 가정: {response:{body:{items:{item:[...]}}}}
        items = (
            data.get("response", {})
                .get("body", {})
                .get("items", {})
                .get("item", [])
        )
        if not items:
            break
        all_items.extend(items)
        snap = raw_dir / f"kto_page{page_no}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        snap.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        if len(items) < num_rows:
            break
        page_no += 1

    if not all_items:
        print("[KTO] 데이터 없음!")
        return pd.DataFrame()

    df = pd.DataFrame(all_items)

    # 표준 컬럼 추정/정규화: 이름/시각/예측치
    # 공공데이터 컬럼명이 고정이 아닐 수 있어 후보군으로 탐색
    def pick(colnames: List[str]) -> Optional[str]:
        for c in colnames:
            if c in df.columns:
                return c
        return None

    name_col = pick(["tAtsNm", "관광지명", "name", "title"])
    ts_col   = pick(["baseYmd", "baseDe", "ymd", "date", "obsrDe"])  # 날짜(YYYYMMDD or YYYY-MM-DD)
    pred_col = pick(["cnctrRate", "cnctrRatePred", "visitorRate", "predicted", "value"])

    std = pd.DataFrame()
    if name_col:
        std["place_name_src"] = df[name_col].astype(str)
    if ts_col:
        # YYYYMMDD → YYYY-MM-DD 변환 시도
        ts_val = pd.to_datetime(df[ts_col].astype(str), errors="coerce")
        std["date"] = ts_val.dt.date
    if pred_col:
        std["visitor_rate_pred"] = pd.to_numeric(df[pred_col], errors="coerce")

    # 해수욕장만 필터링하려면 이름 정규화 후 '해수욕장' 키워드 또는 매핑 사용
    def normalize_beach_name(x: str) -> str:
        x0 = re.sub(r"\s+", "", x)
        x0 = x0.replace("해수욕장", "")
        return x0

    if "place_name_src" in std:
        std["beach_name_std"] = std["place_name_src"].map(normalize_beach_name)
        std["beach_id"] = std["beach_name_std"].str.lower()

    std = std.dropna(subset=["date"]).copy()

    interim_path = interim_dir / f"kto_{area_cd}_{datetime.now().strftime('%Y%m%d')}.csv"
    std.to_csv(interim_path, index=False, encoding="utf-8-sig")
    print(f"[KTO] 표준화 저장: {interim_path}")
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
    base = settings["sources"]["kma"]["base"]
    path = settings["sources"]["kma"]["path"]
    data_type = settings["sources"]["kma"]["data_type"]
    data_cd = settings["sources"]["kma"]["data_cd"]
    date_cd = settings["sources"]["kma"]["date_cd"]

    raw_dir = Path(settings["paths"]["raw_dir"]) / "kma"
    interim_dir = Path(settings["paths"]["interim_dir"]) / "kma"
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    # 월별 청크로 끊어서 호출(일자료는 한 번에 많이 요청해도 되지만 보수적으로 운용)
    def month_chunks(start: date, end: date):
        cur = date(start.year, start.month, 1)
        while cur <= end:
            if cur.month == 12:
                nxt = date(cur.year + 1, 1, 1)
            else:
                nxt = date(cur.year, cur.month + 1, 1)
            chunk_end = min(end, nxt - timedelta(days=1))
            yield cur, chunk_end
            cur = nxt

    all_rows: List[pd.DataFrame] = []
    for stn in stn_ids:
        for mstart, mend in month_chunks(start, end):
            url = base + path
            params = {
                "serviceKey": service_key,
                "pageNo": 1,
                "numOfRows": 500,  # 일자료 월 31일 가정
                "dataType": data_type,
                "dataCd": data_cd,
                "dateCd": date_cd,
                "startDt": mstart.strftime("%Y%m%d"),
                "endDt": mend.strftime("%Y%m%d"),
                "stnIds": stn,
            }
            data = http_get_json(url, params, settings)
            items = (
                data.get("response", {})
                    .get("body", {})
                    .get("items", {})
                    .get("item", [])
            )
            snap = raw_dir / f"asos_{stn}_{mstart.strftime('%Y%m')}.json"
            snap.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

            if not items:
                continue

            df = pd.DataFrame(items)
            # 컬럼 후보 매핑
            # 날짜: tm(YYYY-MM-DD) 또는 ymd
            # 평균기온: avgTa, 강수량합계: sumRn, 평균풍속: avgWs, 평균상대습도: avgRhm 등
            def pick(colnames: List[str]) -> Optional[str]:
                for c in colnames:
                    if c in df.columns:
                        return c
                return None

            tm_col   = pick(["tm", "ymd", "date"])
            tavg_col = pick(["avgTa", "taAvg", "tavg"])
            tmin_col = pick(["minTa", "tmin"])
            tmax_col = pick(["maxTa", "tmax"])
            rain_col = pick(["sumRn", "rnDay", "rain_sum"])
            wind_col = pick(["avgWs", "windAvg"])
            humi_col = pick(["avgRhm", "rhmAvg", "humid_avg"])

            std = pd.DataFrame()
            std["asos_stn_id"] = stn
            if tm_col:
                std["date"] = pd.to_datetime(df[tm_col], errors="coerce").dt.date
            if tavg_col:
                std["tavg"] = pd.to_numeric(df[tavg_col], errors="coerce")
            if tmin_col:
                std["tmin"] = pd.to_numeric(df[tmin_col], errors="coerce")
            if tmax_col:
                std["tmax"] = pd.to_numeric(df[tmax_col], errors="coerce")
            if rain_col:
                std["rain_sum"] = pd.to_numeric(df[rain_col], errors="coerce")
            if wind_col:
                std["wind_avg"] = pd.to_numeric(df[wind_col], errors="coerce")
            if humi_col:
                std["humid_avg"] = pd.to_numeric(df[humi_col], errors="coerce")

            all_rows.append(std)

    if not all_rows:
        print("[KMA] 데이터 없음!")
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    interim_path = interim_dir / f"asos_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
    out.to_csv(interim_path, index=False, encoding="utf-8-sig")
    print(f"[KMA] 표준화 저장: {interim_path}")
    return out

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
                         od_df: pd.DataFrame,
                         kto_df: pd.DataFrame,
                         asos_df: pd.DataFrame,
                         area_cd: str) -> pd.DataFrame:
    # 해변 메타에 ASOS 매핑 부여
    od_map = attach_asos_to_beach(od_df, DEFAULT_BEACH_TO_ASOS)

    # KTO는 이미 beach_id/date/visitor_rate_pred 보유
    left = kto_df.dropna(subset=["beach_id", "date"]).copy()

    # odcloud와 조인(면적 등 메타)
    left = left.merge(
        od_map[["beach_id", "beach_name_std", "sand_area_m2", "lat", "lon", "asos_stn_id"]],
        on="beach_id", how="left"
    )

    # ASOS는 stn_id+date 기준으로 매칭 → 먼저 beach별로 stn_id를 가져오고 date로 조인
    if "asos_stn_id" in left.columns and not asos_df.empty:
        asos_small = asos_df.dropna(subset=["asos_stn_id", "date"]).copy()
        left = left.merge(
            asos_small,
            on=["asos_stn_id", "date"],
            how="left",
            suffixes=("", "_asos")
        )

    # 시간 파생 피처
    peak_hours = settings["features"].get("peak_hours", [])
    left = add_time_features(left, date_col="date", hour_col=None, peak_hours=peak_hours)

    # 학습셋 최소 필수 컬럼 정리
    cols = [
        "beach_id", "beach_name_std", "date",
        "sand_area_m2", "lat", "lon",
        "tavg", "tmin", "tmax", "rain_sum", "wind_avg", "humid_avg",
        "year", "month", "day", "dow", "is_weekend", "season",
        "visitor_rate_pred"
    ]
    for c in cols:
        if c not in left.columns:
            left[c] = pd.NA

    left = left[cols].drop_duplicates()

    # 저장
    processed_dir = Path(settings["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out = processed_dir / f"train_dataset_area{area_cd}_{datetime.now().strftime('%Y%m%d')}.csv"
    left.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[JOIN] 최종 학습셋 저장: {out}")
    return left

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
        # ODCLOUD
        od_dir = Path(settings["paths"]["interim_dir"]) / "odcloud"
        od_files = sorted(od_dir.glob("beach_meta_*.csv"))
        if not od_files:
            print("[JOIN] odcloud 표준화 파일이 없습니다! 먼저 python pipeline.py odcloud 실행!", file=sys.stderr)
            sys.exit(1)
        od_df = pd.read_csv(od_files[-1], encoding="utf-8")

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
