"""CSV ingestion and per-problem feature handling."""
from io import BytesIO
from urllib.parse import urlsplit
import ipaddress
import socket
import pandas as pd
import requests

MAX_BYTES = 12 * 1024 * 1024
MAX_ROWS = 50000
MAX_COLUMNS = 100


def read_csv_bytes(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError('CSV 超過 12 MB 上限')
    try:
        frame = pd.read_csv(BytesIO(raw), low_memory=False)
    except UnicodeDecodeError:
        frame = pd.read_csv(BytesIO(raw), encoding='utf-8-sig', low_memory=False)
    if not 40 <= len(frame) <= MAX_ROWS:
        raise ValueError(f'資料筆數必須介於 40～{MAX_ROWS:,} 筆')
    if not 2 <= len(frame.columns) <= MAX_COLUMNS:
        raise ValueError('CSV 欄位數必須介於 2～100 欄')
    if frame.columns.duplicated().any():
        raise ValueError('CSV 包含重複的欄位名稱')
    return frame


def read_public_https_csv(url):
    """Accept public HTTPS CSV only. Do not follow redirects or local/private DNS."""
    parsed = urlsplit(url.strip())
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.port not in (None,443):
        raise ValueError('請提供公開 HTTPS CSV 直連網址（不支援帳密、Port 或重新導向）')
    host = parsed.hostname
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError('無法解析 CSV 網址網域') from exc
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError('不可連接內網、本機或非公開網域')
    try:
        response = requests.get(url, timeout=(5, 15), allow_redirects=False,
                                headers={'User-Agent': 'HW2-Regression-Project/1.0'}, stream=True)
        if 300 <= response.status_code < 400:
            raise ValueError('此 URL 會重新導向；請貼上最終的 CSV 直連網址')
        response.raise_for_status()
        data = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise ValueError('遠端 CSV 超過 12 MB 上限')
        return bytes(data)
    except requests.RequestException as exc:
        raise ValueError(f'下載 CSV 失敗：{exc}') from exc


def prepare_dataset(df, problem, target, removed_columns=None):
    data = df.copy()
    if problem == 'King County House Sales':
        # Exclude ID (not a predictor); convert sale date into calendar features.
        data = data.drop(columns=['id'], errors='ignore')
        if 'date' in data.columns:
            date_text = data['date'].astype(str).str.slice(0, 8)
            parsed = pd.to_datetime(date_text, format='%Y%m%d', errors='coerce')
            if parsed.isna().all():
                parsed = pd.to_datetime(data['date'], errors='coerce')
            data['sale_year'] = parsed.dt.year.astype(float)
            data['sale_month'] = parsed.dt.month.astype(float)
            data = data.drop(columns=['date'])
        if 'zipcode' in data.columns:
            data['zipcode'] = data['zipcode'].astype('string').fillna('Unknown').astype(str)
    for col in removed_columns or []:
        if col != target:
            data = data.drop(columns=[col], errors='ignore')
    if target not in data.columns:
        raise ValueError(f'找不到目標欄位：{target}')
    if not pd.api.types.is_numeric_dtype(data[target]):
        data[target] = pd.to_numeric(data[target], errors='coerce')
    if data[target].notna().sum() < 40:
        raise ValueError('有效數值目標值少於 40 筆，無法回歸')
    # Reject high-cardinality text columns: unsafe for dense one-hot encoding.
    bad = [c for c in data.select_dtypes(include=['object','category','string']).columns
           if c != target and data[c].nunique(dropna=True) > 100]
    if bad:
        raise ValueError('以下文字欄位的不同值超過 100，請在「排除欄位」取消選取它們以避免 One-Hot 爆量：' + ', '.join(bad))
    return data
