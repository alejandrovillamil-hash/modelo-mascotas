#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

ARTIFACTS_MODEL_PATH = ARTIFACTS_DIR / "artefactos_algoritmo.pkl"

from sklearn.metrics import mean_absolute_error


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import tempfile
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import os

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import category_encoders as ce
from sklearn.linear_model import HuberRegressor
from sklearn.inspection import permutation_importance

import re
from sklearn.base import BaseEstimator, TransformerMixin
from rapidfuzz import process, fuzz

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import get_as_dataframe, set_with_dataframe
'''
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.oauth2.service_account import Credentials
import json
'''
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import traceback

fecha_str = datetime.now().strftime("%Y%m%d_%H%M%S")
sns.set_style('whitegrid')
RND = 42
'''
INPUT_DIR = Path('C:/Users/alejo/OneDrive/Escritorio/Modelo mascotas/input/')
OUTPUT_DIR = Path('C:/Users/alejo/OneDrive/Escritorio/Modelo mascotas/output/')

AUTH_PATH = Path(INPUT_DIR / 'automatizacion-480622-83f05856e13e.json')

ARTIFACTS_MODEL_PATH = Path(OUTPUT_DIR / "artefactos_algoritmo.pkl")
ARTIFACTS_DATA_ENG_PATH = Path(OUTPUT_DIR / "artefactos_data_engineering.pkl")
'''
PARAMETRIZACION_SHEET = "parametrizacion"
EJECUCION_SHEET = "ejecucion"
'''
PETTITO_SHEET = "Master Puppy Information Sheet"
MASTER_WORKSHEET = "Master Tab"
MIRROR_WORKSHEET = "Mirror Master Tab"
'''
PETTITO_SHEET = "datos_mascotas"
MASTER_WORKSHEET = "Hoja1"
MIRROR_WORKSHEET = "Hoja2"

# TRAIN = False
#NEW_RECORDS = False
#WORKSHEET_NUEW_RECORDS = "Hoja5"

'''
SMOOTH_TE = 1.0
USE_KFOLD_TE = True
N_SPLITS_TE = 5
RND = 42
TE_ENCODERS = {}

HORIZON = 14
LEAD_TIME = 7
SAFETY = 0.20
TOP_K = 10

DEFAULT_FLOOR_PCT = 0.00
def enviar_correo(asunto, mensaje):DEFAULT_MARGIN = 0.00'''

#TRAIN = os.getenv("TRAIN", "false").lower() == "true"

def str2bool(v):
    return str(v).lower() in ("true", "1", "yes", "y")

TRAIN = str2bool(os.getenv("TRAIN", "false"))
NEW_RECORDS = str2bool(os.getenv("NUEVO", "false"))
WORKSHEET_NUEW_RECORDS = os.getenv("NOMBRE_NUEVOS")


print("▶ TRAIN:", TRAIN)
print("▶ SHEET_ID:", SHEET_ID)
print("▶ NOTIFY_EMAIL:", NOTIFY_EMAIL)



def main():

    


    # In[2]:


    def enviar_correo(asunto, mensaje):
        EMAIL_CONFIG = {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "alejandrovillamil@pettitousa.com",
            "sender_password": "eryw hgrh wmli whxy",
            "receiver_email": [
                "alejandrovillamil@pettitousa.com"
                # puedes agregar más correos aquí
            ]
        }

        try:
            msg = MIMEMultipart()
            msg["From"] = EMAIL_CONFIG["sender_email"]
            msg["To"] = ", ".join(EMAIL_CONFIG["receiver_email"])
            msg["Subject"] = asunto

            msg.attach(MIMEText(mensaje, "plain"))

            server = smtplib.SMTP(
                EMAIL_CONFIG["smtp_server"],
                EMAIL_CONFIG["smtp_port"]
            )
            server.starttls()
            server.login(
                EMAIL_CONFIG["sender_email"],
                EMAIL_CONFIG["sender_password"]
            )
            server.send_message(msg)
            server.quit()

            print("📧 Correo enviado correctamente")

        except Exception as e:
            print("❌ Error enviando correo:", e)


    # In[3]:


    # Google Sheet API

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "service_account.json")

    creds = ServiceAccountCredentials.from_json_keyfile_name(
        creds_path, scope
    )

    client = gspread.authorize(creds)

    sheet = client.open(PETTITO_SHEET)
    sheet_parametrizacion = client.open(PARAMETRIZACION_SHEET)
    sheet_ejecucion = client.open(EJECUCION_SHEET)

    # Hojas
    sheet_master = sheet.worksheet(MASTER_WORKSHEET)
    worksheet_espejo_actual = sheet.worksheet(MIRROR_WORKSHEET)

    # Funciones auxiliares
    def get_sheet_df(worksheet):
        """Convierte una worksheet a DataFrame limpio"""
        df = get_as_dataframe(worksheet, dtype=str).dropna(how="all")
        df = df.loc[:, df.columns.notna()]
        df = df.loc[:, df.columns.str.strip() != ""]
        df.columns = df.columns.str.strip()
        return df
        
    def get_price_columns(cols):
        """Devuelve columnas Price_XX ordenadas por número"""
        return sorted(
            [c for c in cols if c.startswith("Price_")],
            key=lambda x: int(x.split("_")[1])
        )

    def fusionar_historico_precios_updates(df_old, df_new):
        """
        Genera updates y nuevas columnas Price/Date para fusionar df_old -> df_new
        """
        df_new = df_new.copy()
        updates = []
        new_columns = []

        price_cols_old = get_price_columns(df_old.columns)

        for _, old_row in df_old.iterrows():
            chip = str(old_row["Microchip #"]).strip()
            match = df_new[df_new["Microchip #"].astype(str).str.strip() == chip]
            if match.empty:
                continue

            idx_new = match.index[0]

            for last_price_col in price_cols_old:
                last_date_col = last_price_col.replace("Price_", "Date_")
                last_price_old = old_row.get(last_price_col)
                last_date_old  = old_row.get(last_date_col)

                # reemplazar NaN/None por ""
                if pd.isna(last_price_old) or last_price_old is None:
                    last_price_old = ""
                if pd.isna(last_date_old) or last_date_old is None:
                    last_date_old = ""

                # agregar columnas nuevas si no existen
                if last_price_col not in df_new.columns:
                    df_new[last_price_col] = ""
                    new_columns.append(last_price_col)
                if last_date_col not in df_new.columns:
                    df_new[last_date_col] = ""
                    new_columns.append(last_date_col)

                # preparar updates
                updates.append({"row": idx_new + 2, "col_name": last_price_col, "value": last_price_old})
                updates.append({"row": idx_new + 2, "col_name": last_date_col,  "value": last_date_old})

        return updates, list(set(new_columns))

    def map_headers_to_cols(worksheet):
        """Mapear nombre de columna a índice de columna en Sheet"""
        headers = worksheet.row_values(1)
        return {h: i+1 for i, h in enumerate(headers)}

    def safe_value(val):
        """Evitar errores de NaN en Google Sheets"""
        if val is None:
            return ""
        if isinstance(val, float) and math.isnan(val):
            return ""
        return val

    def ordenar_price_date_cols(cols):
        """Ordena columnas Price_XX y Date_XX intercaladas"""
        price_cols = sorted([c for c in cols if c.startswith("Price_")], key=lambda x: int(x.split("_")[1]))
        date_cols  = sorted([c for c in cols if c.startswith("Date_")], key=lambda x: int(x.split("_")[1]))
        ordered = []
        for p in price_cols:
            ordered.append(p)
            d = p.replace("Price_", "Date_")
            if d in date_cols:
                ordered.append(d)
        other_cols = [c for c in cols if not (c.startswith("Price_") or c.startswith("Date_"))]
        return other_cols + ordered

    # Leer datos
    df_old = get_sheet_df(worksheet_espejo_actual)

    worksheet_ejecucion_backup = sheet_ejecucion.add_worksheet(title=f'backup_mirror_{fecha_str}', rows="10000", cols="50")
    df_old = df_old.fillna("")
    datos = df_old.values.tolist()
    encabezados = df_old.columns.tolist()
    worksheet_ejecucion_backup.update('A1', [encabezados] + datos)

    worksheet_a_eliminar = sheet.worksheet(MIRROR_WORKSHEET)
    sheet.del_worksheet(worksheet_a_eliminar)

    # Duplicar Hoja1 → Hoja3 (a la derecha de la master)
    index_master = sheet_master.index
    sheet.duplicate_sheet(
        source_sheet_id=sheet_master.id,
        new_sheet_name=MIRROR_WORKSHEET,
        insert_sheet_index=index_master + 1
    )
    worksheet_espejo_nueva = sheet.worksheet(MIRROR_WORKSHEET)
    df_new = get_sheet_df(worksheet_espejo_nueva)

    # Fusionar precios históricos
    updates, new_columns = fusionar_historico_precios_updates(df_old, df_new)

    header_map = map_headers_to_cols(worksheet_espejo_nueva)

    # Crear columnas nuevas si es necesario
    if new_columns:
        worksheet_espejo_nueva.add_cols(len(new_columns))
        headers = worksheet_espejo_nueva.row_values(1)
        for col in new_columns:
            headers.append(col)
        # Reordenar columnas Price/Date
        headers = ordenar_price_date_cols(headers)
        worksheet_espejo_nueva.update('1:1', [headers])
        header_map = map_headers_to_cols(worksheet_espejo_nueva)

    # Crear celdas para actualizar
    cell_list = [
        gspread.Cell(u["row"], header_map[u["col_name"]], safe_value(u["value"]))
        for u in updates
    ]

    # Actualizar por lotes
    if cell_list:
        worksheet_espejo_nueva.update_cells(cell_list)

    worksheet = sheet.worksheet(MIRROR_WORKSHEET)
    df_sheet = get_as_dataframe(worksheet, dtype=str).dropna(how="all")


    # In[4]:


    # Data engineering

    if NEW_RECORDS:
        worksheet_new = sheet.worksheet(WORKSHEET_NUEW_RECORDS)
        df_sheet_new = get_as_dataframe(worksheet_new, dtype=str).dropna(how="all")

        df_new_records = df_sheet_new.copy()
        df_new_records.columns = df_new_records.columns.str.strip()
        df_new_records = df_new_records[[
            "Microchip#", "Breed", "Color", "Sex  M|F", "Breeder", "Price",
            "Arrival Date", "Sold date", "Location", "Idle Days",
            "Size", "Variety", "Recibio Registro? SI/NO"
        ]]
        df_new_records["Status"] = ""
        df_new_records.columns = [
            "id", "raza", "color", "genero", "criador", "precio_venta",
            "fecha_listado", "fecha_venta", "tienda", "dias_en_tienda",
            "tamano", "caracteristica", "registro", "estado"
        ]

    df = df_sheet.copy()
    # Parametrizacion
    worksheet_base_piso = sheet_parametrizacion.worksheet('base_piso')
    df_base_piso = get_sheet_df(worksheet_base_piso)
    worksheet_estandarizacion = sheet_parametrizacion.worksheet('estandarizacion')
    df_estandarizacion = get_sheet_df(worksheet_estandarizacion)
    worksheet_estandarizacion_diccionario = sheet_parametrizacion.worksheet('estandarizacion_diccionario')
    df_estandarizacion_diccionario = get_sheet_df(worksheet_estandarizacion_diccionario)
    worksheet_descuento_idle_days = sheet_parametrizacion.worksheet('descuento_idle_days')
    df_descuento_idle_days = get_as_dataframe(worksheet_descuento_idle_days, dtype=str).dropna(how="all")

    print("df:", df.shape, "worksheet_base_piso:", df_base_piso.shape, "est:", df_estandarizacion.shape, "dic-est:", df_estandarizacion_diccionario.shape, "disc-rules:", df_descuento_idle_days.shape)

    df.columns = df.columns.str.strip()
    df = df[[
        "Microchip #", "Breed", "Color", "Sex  M|F", "Breeder", "Price",
        "Arrival Date", "Sold date", "Location", "Idle Days",
        "Size", "Variety", "Recibio Registro? SI/NO", "Status"
    ]]

    df.columns = [
        "id", "raza", "color", "genero", "criador", "precio_venta",
        "fecha_listado", "fecha_venta", "tienda", "dias_en_tienda",
        "tamano", "caracteristica", "registro", "estado"
    ]

    if NEW_RECORDS:
        df_new_records['source'] = 1
        df['source'] = 0
        df = pd.concat([df, df_new_records], ignore_index=True)

    df['comentario'] = ''

    required = ['id','raza','color','genero']
    df = df.dropna(subset=required, how='any').copy()

    # Mover valores
    valores_caracteristica = ["toy", "micro", "teacup", "mini"]
    valores_tamano = ["moyan", "neutered"]

    mask_caracteristica = df['caracteristica'].str.lower().isin(valores_caracteristica)
    df.loc[mask_caracteristica, 'tamano'] = df.loc[mask_caracteristica, 'caracteristica']
    df.loc[mask_caracteristica, 'caracteristica'] = np.nan

    mask_tamano = df['tamano'].str.lower().isin(valores_tamano)
    df.loc[mask_tamano, 'caracteristica'] = df.loc[mask_tamano, 'tamano']
    df.loc[mask_tamano, 'tamano'] = np.nan

    df.loc[df['color'].str.contains('tri', case=False, na=False), 'color'] = 'tricolor'

    CANONICOS = {
        'raza': sorted(df_estandarizacion['raza'].dropna().astype(str).str.strip().str.lower().unique()),
        'color': sorted(df_estandarizacion['color'].dropna().astype(str).str.strip().str.lower().unique()),
        'tienda': sorted(df_estandarizacion['tienda'].dropna().astype(str).str.strip().str.lower().unique()),
        'tamano': sorted(df_estandarizacion['tamano'].dropna().astype(str).str.strip().str.lower().unique()),
        'caracteristica': sorted(df_estandarizacion['caracteristica'].dropna().astype(str).str.strip().str.lower().unique()),
        'registro': sorted(df_estandarizacion['registro'].dropna().astype(str).str.strip().str.lower().unique())
    }

    def clean_basic(s):
        if pd.isna(s): return ""
        s = str(s).strip().lower()
        s = s.replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n')
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'[^a-z0-9\s&\-\+\(\)]','',s)
        return s

    df_clean = df.copy()
    for c in ['raza','color','tienda','tamano','caracteristica','registro']:
        df_clean[c] = df_clean[c].astype(str).apply(clean_basic).replace({'nan': np.nan})

    AUTO_SINONIMOS = { 'raza':{}, 'color':{}, 'tienda':{}, 'tamano':{}, 'caracteristica':{}, 'registro':{} }

    for campo in ['raza','color','tienda','tamano','caracteristica','registro']:
        AUTO_SINONIMOS[campo] = {canon:[canon] for canon in CANONICOS[campo]}
        valores = df_clean[campo].dropna().unique().tolist()
        for v in valores:
            match, score, _ = process.extractOne(v, CANONICOS[campo], scorer=fuzz.token_sort_ratio)
            if score >= 85:
                # agregar como sinónimo
                if v not in AUTO_SINONIMOS[campo][match]:
                    AUTO_SINONIMOS[campo][match].append(v)
            else:
                # valor raro → no se asigna, quedará 'desconocido'
                pass

    print("Mapas generados automáticamente:")

    for campo, mp in AUTO_SINONIMOS.items():
        print(" -", campo, ":", len(mp), "canónicos")

    class SemanticCleaner(BaseEstimator, TransformerMixin):
        def __init__(self, mapping_dicts, min_freq=0.002, use_fuzzy=True, fuzzy_threshold=85):
            """
            mapping_dicts : dict(col -> {canonico: [sinonimos...]})
            min_freq      : puede ser float fijo o dict(col -> float)
            """
            self.mapping_dicts = mapping_dicts
            self.min_freq = min_freq
            self.use_fuzzy = use_fuzzy
            self.fuzzy_threshold = fuzzy_threshold
            
            # auditoría
            self.audit = {k: [] for k in mapping_dicts.keys()}

        def _clean(self, s):
            if pd.isna(s):
                return np.nan
            s = str(s).strip().lower()
            s = s.replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ñ','n')
            s = re.sub(r'\s+', ' ', s)
            s = re.sub(r'[^a-z0-9\s&\-\+\(\)]', '', s)
            return s

        def fit(self, X, y=None):
            self.canon_lists = {col: list(self.mapping_dicts[col].keys()) for col in X.columns}
            return self

        def transform(self, X):
            X = X.copy()
        
            for col in self.mapping_dicts.keys():
                
                inv = {}
                for canon, sins in self.mapping_dicts[col].items():
                    for s in sins:
                        inv[s] = canon
        
                def map_one(v):
                    if pd.isna(v):
                        return np.nan
        
                    v_clean = self._clean(v)
        
                    if v_clean in inv:
                        self.audit[col].append((v_clean, inv[v_clean], "exact"))
                        return inv[v_clean]
        
                    if self.use_fuzzy:
                        match, score, _ = process.extractOne(
                            v_clean, self.canon_lists[col], scorer=fuzz.token_sort_ratio
                        )
        
                        if score >= self.fuzzy_threshold:
                            self.audit[col].append((v_clean, match, f"fuzzy@{score}"))
                            return match
                        else:
                            self.audit[col].append((v_clean, match, f"lowconf@{score}"))
                            return np.nan
        
                    return np.nan
        
                # limpieza + mapeo
                X[col] = X[col].apply(map_one)
        
                if col == "raza":
                    X["raza_original"] = X[col]

                if col == "raza":
                    X["raza_original"] = X[col]
                    X.loc[X["raza_original"] == "standard goldendoodle", "raza_original"] = "goldendoodle"
        
                # threshold
                if isinstance(self.min_freq, dict):
                    freq_threshold = self.min_freq.get(col, 0.002)
                else:
                    freq_threshold = self.min_freq
        
                freqs = X[col].value_counts(normalize=True)
                low_levels = freqs[freqs < freq_threshold].index.tolist()
        
                X[col] = X[col].replace(low_levels, 'otros').fillna('desconocido')
        
            return X

    for _, row in df_estandarizacion_diccionario.iterrows():
        col = row["columna"]
        canon = row["canonico"]
        sin = row["sinonimo"]
        AUTO_SINONIMOS[col][canon].append(sin)

    df_base_piso['raza'] = df_base_piso['raza'].astype(str).str.strip().str.lower()
    df['raza'] = df['raza'].astype(str).str.strip().str.lower()
    cols = ['raza','color','tienda','tamano','caracteristica','registro']

    sc = SemanticCleaner(
        mapping_dicts=AUTO_SINONIMOS,
        min_freq={
            'raza':0.002,
            'color':0.002,
            'tienda':0.002,
            'tamano':0,
            'caracteristica':0,
            'registro':0.002
        }
    )
    sc.fit(df[cols])
    df = sc.transform(df)

    df = df.merge(
        df_base_piso[['raza','raza_categoria','base','min']].drop_duplicates('raza'),
        left_on='raza_original',
        right_on='raza',
        how='left'
    )
    df.drop(columns=['raza_y'], inplace=True)
    df.rename(columns={'raza_x': 'raza'}, inplace=True)

    audit_rows = []
    for col, logs in sc.audit.items():
        for ori, mapped, tag in logs:
            if tag.startswith("lowconf"):
                audit_rows.append({"col":col,"original":ori,"mapped_candidate":mapped,"tag":tag})
    audit_df = pd.DataFrame(audit_rows).drop_duplicates()

    try:
        worksheet_auditoria_estandarizacion = sheet_parametrizacion.worksheet('auditoria_estandarizacion')
        sheet_parametrizacion.del_worksheet(worksheet_auditoria_estandarizacion)
    except gspread.exceptions.WorksheetNotFound:
        print(f"La hoja 'auditoria_estandarizacion' no existe. Se creará una nueva.")

    worksheet_auditoria_estandarizacion = sheet_parametrizacion.add_worksheet(title='auditoria_estandarizacion', rows="200", cols="20")

    audit_df = audit_df.fillna("")
    datos = audit_df.values.tolist()
    encabezados = audit_df.columns.tolist()

    worksheet_auditoria_estandarizacion.update('A1', [encabezados] + datos)

    mask_missing = df['raza_categoria'].isna()

    df['fecha_listado'] = pd.to_datetime(df['fecha_listado'], errors='coerce')
    df['fecha_venta']   = pd.to_datetime(df['fecha_venta'],   errors='coerce')
    df['vendido_flag'] = df['fecha_venta'].notna().astype(int)
    df['precio_venta'] = pd.to_numeric(df['precio_venta'].str.replace(r'[$,]', '', regex=True),errors='coerce')

    ref_date = pd.Timestamp.now()
    #ref_date = pd.Timestamp('2025-12-30')
    df['dias_en_tienda'] = (df['fecha_venta'].fillna(ref_date) - df['fecha_listado']).dt.days.clip(lower=0)

    agg_raza = df[df['vendido_flag']==1].groupby('raza').agg(
        med_dias=('dias_en_tienda','median'),
        med_precio=('precio_venta','median'),
        n_raza=('id','count')
    ).reset_index()

    agg_raza['rotacion'] = agg_raza['n_raza'] / agg_raza['med_dias'].replace(0,np.nan)
    agg_raza['rotacion'] = agg_raza['rotacion'].fillna(0)

    q_d = agg_raza['med_dias'].quantile([0.1,0.3,0.5,0.7,0.9]).values
    q_p = agg_raza['med_precio'].quantile([0.1,0.3,0.5,0.7,0.9]).values
    q_r = agg_raza['rotacion'].quantile([0.1,0.3,0.5,0.7,0.9]).values
    q_n = agg_raza['n_raza'].quantile([0.1,0.3,0.5,0.7,0.9]).values

    def map_cat(d,p,r,n):
        try:
            if d<=q_d[0] and p>=q_p[4] and r>=q_r[4] and n>=q_n[4]: return 'A+'
            if d<=q_d[1] and p>=q_p[3] and r>=q_r[3] and n>=q_n[3]: return 'A'
            if d<=q_d[2] and p>=q_p[2] and r>=q_r[2] and n>=q_n[2]: return 'B'
            if d<=q_d[3] and p<=q_p[1] and r>=q_r[1] and n>=q_n[1]: return 'C'
            if d<=q_d[4] and p<=q_p[0] and r>=q_r[0] and n>=q_n[0]: return 'D'
            if d> q_d[4] and p<=q_p[0] and r< q_r[0] and n< q_n[0]: return 'E'
        except:
            pass
        return 'F'

    agg_raza['raza_categoria_advanced'] = agg_raza.apply(
        lambda row: map_cat(row['med_dias'], row['med_precio'], row['rotacion'], row['n_raza']),
        axis=1
    )

    for c in ['med_precio', 'n_raza', 'rotacion']:
        if c not in df.columns:
            df[c] = np.nan

    if mask_missing.any():
        df_missing = df.loc[mask_missing,['raza']].drop_duplicates().merge(
            agg_raza[['raza','raza_categoria_advanced','med_precio','n_raza','rotacion']],
            on='raza',
            how='left'
        )
        for _, r in df_missing.iterrows():
            cond = (df['raza']==r['raza']) & (df['raza_categoria'].isna())
            df.loc[cond,'raza_categoria'] = r['raza_categoria_advanced']
            df.loc[cond,'med_precio']     = r['med_precio']
            df.loc[cond,'n_raza']         = r['n_raza']
            df.loc[cond,'rotacion']       = r['rotacion']
            df.loc[cond,'comentario']     = "No se encontro raza en archivo 'raza_base_piso'"

    df['raza_categoria'] = df['raza_categoria'].fillna('F')
    df['med_precio']     = df['med_precio'].fillna(df['med_precio'].median())
    df['n_raza']         = df['n_raza'].fillna(0)
    df['rotacion']       = df['rotacion'].fillna(0)

    agg_t = df.groupby('tienda').agg(
        precio_mean_tienda=('precio_venta','mean'),
        ventas_total=('vendido_flag','sum'),
        total_listados=('id','count')
    ).reset_index()

    agg_t['prob_vender_tienda'] = agg_t['ventas_total'] / agg_t['total_listados']
    df = df.merge(agg_t[['tienda','precio_mean_tienda','prob_vender_tienda']], on='tienda', how='left')

    df['raza_x_tienda'] = df['raza'] + '___' + df['tienda']
    df['raza_x_color']  = df['raza'] + '___' + df['color']

    for col in ['raza_x_tienda','raza_x_color']:
        freq = df[col].value_counts(normalize=True)
        low = freq[freq < 0.001].index
        df[col] = df[col].replace(low, 'otros')

    df['estado'] = df['estado'].astype(str).str.strip().str.lower()

    if NEW_RECORDS:
        predict_df = df[(df['source'] == 1)].copy().reset_index(drop=True)
    else:
        predict_df = df[(df['vendido_flag'] == 0) & (df['estado'] != 'inactive')].copy().reset_index(drop=True)

    print("Completado df:", df.shape, " predict_df:", predict_df.shape)


    # In[5]:


    # Inventario tiendas

    df_filtered = df[(df['tienda'] != 'otros') & (df['raza'] != 'otros') & (df['color'] != 'otros')]

    combo = df_filtered.groupby(['tienda','raza','color']).agg(
        n_listados=('id','count'),
        n_vendidos=('vendido_flag','sum'),
        precio_promedio=('precio_venta','mean'),
        dias_medianos=('dias_en_tienda','median'),
        prob_mean=('vendido_flag','mean')
    ).reset_index()

    combo['dias_medianos'] = combo['dias_medianos'].replace(0, 1)
    combo['rotacion'] = combo['n_vendidos'] / combo['dias_medianos']
    combo['ventas_esperadas_14d'] = combo['prob_mean'] * combo['n_listados']

    def normalize(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    combo['prob_norm'] = normalize(combo['prob_mean'])
    combo['rot_norm'] = normalize(combo['rotacion'])
    combo['precio_norm'] = normalize(combo['precio_promedio'])

    combo['score'] = (
        0.35 * combo['prob_norm'] +
        0.30 * combo['rot_norm'] +
        0.20 * combo['precio_norm'] +
        0.10 * np.log1p(combo['n_listados']) +
        0.05 * (1 / (1 + combo['dias_medianos']))
    )

    HORIZON = 14
    LEAD_TIME = 7
    SAFETY = 0.20
    TOP_K = 10

    combo['inventario_base'] = combo['ventas_esperadas_14d'] * (LEAD_TIME / HORIZON)
    combo['safety_stock'] = combo['inventario_base'] * SAFETY
    combo['inventario_ideal'] = np.ceil(combo['inventario_base'] + combo['safety_stock']).astype(int)

    combo_sorted = combo.sort_values(
        ['tienda','score'],
        ascending=[True, False]
    ).reset_index(drop=True)

    combo_topk = combo_sorted.groupby('tienda').head(TOP_K).reset_index(drop=True)

    try:
        worksheet_inventario_tiendas = sheet_parametrizacion.worksheet('inventario_tiendas')
        sheet_parametrizacion.del_worksheet(worksheet_inventario_tiendas)
    except gspread.exceptions.WorksheetNotFound:
        print(f"La hoja 'inventario_tiendas' no existe. Se creará una nueva.")

    worksheet_inventario_tiendas = sheet_parametrizacion.add_worksheet(title='inventario_tiendas', rows="200", cols="20")

    combo_topk = combo_topk.fillna("")
    datos = combo_topk.values.tolist()
    encabezados = combo_topk.columns.tolist()

    worksheet_inventario_tiendas.update('A1', [encabezados] + datos)


    # In[6]:


    # Target Encoding (KFold + fallback)

    df['precio_log'] = np.log1p(df['precio_venta'].fillna(0))

    categorical_high_card = ['raza', 'color', 'raza_x_tienda', 'raza_x_color']
    te_cols = [f"te_{c}" for c in categorical_high_card]

    for tc in te_cols:
        df[tc] = np.nan

    SMOOTH_TE = 1.0
    USE_KFOLD_TE = True
    N_SPLITS_TE = 5
    RND = 42

    TE_ENCODERS = {}

    # TARGET ENCODING K-FOLD POR COLUMNA
    if USE_KFOLD_TE:
        kf = KFold(n_splits=N_SPLITS_TE, shuffle=True, random_state=RND)

        for col in categorical_high_card:
            te_name = f"te_{col}"
            df[te_name] = np.nan

            # KFold TE (no hay fuga de información --- leakage-free)
            for train_idx, val_idx in kf.split(df):
                tr = df.iloc[train_idx]
                vl = df.iloc[val_idx]

                te = ce.TargetEncoder(cols=[col], smoothing=SMOOTH_TE)
                te.fit(tr[[col]], tr['precio_log'])

                encoded_vals = te.transform(vl[[col]]).values.flatten()
                df.loc[df.index[val_idx], te_name] = encoded_vals

            # Fallback global (garantiza que no queden NaN)
            te_full_col = ce.TargetEncoder(cols=[col], smoothing=SMOOTH_TE)
            te_full_col.fit(df[[col]], df['precio_log'])

            fallback_vals = te_full_col.transform(df[[col]]).values.flatten()
            mask = df[te_name].isna()
            df.loc[mask, te_name] = fallback_vals[mask]

            # Guardar encoder final para inferencia
            TE_ENCODERS[col] = te_full_col

    else:
        # Sin KFold → TE normal
        for col in categorical_high_card:
            te_full_col = ce.TargetEncoder(cols=[col], smoothing=SMOOTH_TE)
            df[f"te_{col}"] = te_full_col.fit_transform(df[[col]], df['precio_log']).values.flatten()
            TE_ENCODERS[col] = te_full_col

    print(f"Target Encoding aplicado correctamente. Columnas generadas: {te_cols}")
    print("Encoders guardados para inferencia:", list(TE_ENCODERS.keys()))


    # In[7]:


    # ColumnTransformer final y features

    numeric_features = ['precio_mean_tienda', 'prob_vender_tienda']
    categorical_low_card = ['tienda', 'genero', 'raza_categoria', 'tamano', 'caracteristica', 'registro']

    num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_low_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
                                   ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    # ColumnTransformer: deja pasar las columnas te_... (remainder='passthrough')
    preproc = ColumnTransformer(transformers=[
        ('num', num_transformer, numeric_features),
        ('catlow', cat_low_transformer, categorical_low_card)
    ], remainder='passthrough', sparse_threshold=0)

    # Lista final de features tal como la verá el modelo (num + ohe_low + te_cols)
    # Para obtener nombres reales después de ajustar preproc, hacemos fit_simple:
    _sample = df.copy()
    _prepped = preproc.fit_transform(_sample)
    # reconstruir los nombres de columnas resultantes
    ohe_cols = list(preproc.named_transformers_['catlow'].named_steps['ohe'].get_feature_names_out(categorical_low_card))
    final_feature_names = numeric_features + ohe_cols + te_cols

    print("Final feature names (ejemplo):", final_feature_names)


    # In[8]:


    # Guardar artefactos data engineering
    '''
    joblib.dump({
        'preproc': preproc,
        'sc': sc,
        'te_encoders': globals().get('TE_ENCODERS', {}),
        'final_feature_names': final_feature_names
    }, ARTIFACTS_DATA_ENG_PATH)
    print("Artefactos guardados en:", ARTIFACTS_DATA_ENG_PATH)
    '''

    # In[9]:


    # Variables de entrenamiento

    TARGET = 'precio_log'
    FEATURES = numeric_features + categorical_low_card + te_cols

    train_df = df[df['vendido_flag']==1].copy()
    X = train_df[FEATURES].copy()
    y = train_df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RND)
    print("Train:", X_train.shape, "Test:", X_test.shape)


    # In[10]:


    # Pipeline + GridSearch

    if TRAIN:
        lgb_model = lgb.LGBMRegressor(
            objective='regression',
            boosting_type='gbdt',
            n_estimators=800,
            random_state=RND,
            n_jobs=-1,
            learning_rate=0.04,
            max_depth=8,
            num_leaves=60,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0
        )
        
        pipe = Pipeline([
            ('preproc', preproc),
            ('model', lgb_model)
        ])
        
        param_grid = {
            'model__num_leaves': [40, 60],
            'model__learning_rate': [0.02, 0.04],
            'model__min_child_samples': [10, 20],
            'model__reg_alpha': [0.3, 0.6],
            'model__reg_lambda': [0.5, 1.0]
        }
        
        cv = KFold(n_splits=5, shuffle=True, random_state=RND)
        
        gs = GridSearchCV(pipe, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=cv, verbose=2, n_jobs=-1)
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        print("Best params:", gs.best_params_)


    # In[11]:


    if TRAIN:
        # Determinar el mejor modelo del GridSearch o RandomizedSearch
        search_obj = gs if 'gs' in locals() else rs
        best_model = search_obj.best_estimator_
        
        # Asegurar que X_test tiene todas las columnas target-encoded
        X_test = train_df.loc[X_test.index, numeric_features + categorical_low_card + te_cols]
        
        # Predicciones en test
        y_pred_log = best_model.predict(X_test)
        y_pred = np.expm1(y_pred_log) if np.any(y_test > 0) else y_pred_log
        y_true = np.expm1(y_test) if np.any(y_test > 0) else y_test
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"\nResultados finales del modelo optimizado:")
        print(f"MAE  = {mae:.3f}")
        print(f"RMSE = {rmse:.3f}")
        print(f"R²   = {r2:.3f}")


    # In[12]:


    # Corrección de sesgo robusta (usando residuales en train con cross_val_predict)

    if TRAIN:   
        # cross-validated predictions on train to model residuals
        cv_preds_log = cross_val_predict(best_model, X_train, y_train, cv=cv, method='predict', n_jobs=-1)
        residuals_train = y_train.values - cv_preds_log  # residuals in log space
        
        # Fit robust regressor (Huber) to predict residual as function of predicted log
        corr_model = HuberRegressor().fit(cv_preds_log.reshape(-1,1), residuals_train)
        
        # Apply correction on test predictions
        y_pred_log_corr = y_pred_log + corr_model.predict(y_pred_log.reshape(-1,1))
        y_pred_corr_orig = np.expm1(y_pred_log_corr)
        
        print("MAE antes corrección:", mean_absolute_error(y_true, y_pred))
        print("MAE después corrección:", mean_absolute_error(y_true, y_pred_corr_orig))


    # In[13]:


    # Importar artefactos

    if not TRAIN:
        if not ARTIFACTS_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No existe el artefacto: {ARTIFACTS_MODEL_PATH}"
            )

        artifacts = joblib.load(ARTIFACTS_MODEL_PATH)

        best_model = artifacts['best_model']
        corr_model = artifacts['corr_model']
        metrics = artifacts['metrics']
        final_feature_names = artifacts['final_feature_names']

        print("✔ Artefactos cargados correctamente")


    # In[14]:


    # Funciones de negocio

    discount_rules = {}

    df_descuento_idle_days.columns = df_descuento_idle_days.columns.map(
        lambda c: "" if pd.isna(c) else str(int(c)) if isinstance(c, float) and c.is_integer() else str(c)
    )

    for index, row in df_descuento_idle_days.iterrows():
        key = row['Unnamed: 0']
        values = [(int(col) if col.isdigit() else col, float(row[col].replace(',', '.'))) for col in df_descuento_idle_days.columns[1:]]
        discount_rules[key] = values

    DEFAULT_FLOOR_PCT = 0.00
    DEFAULT_MARGIN = 0.00

    def preparar_fila_inferencia(perro_row, te_encoders,
                                 numeric_features, categorical_low_card,
                                 categorical_high_card, te_cols):

        if isinstance(perro_row, dict):
            row_df = pd.DataFrame([perro_row])
        else:
            row_df = pd.DataFrame([perro_row.to_dict()])

        # Asegurar todas las columnas necesarias
        for col in numeric_features + categorical_low_card + categorical_high_card:
            if col not in row_df.columns:
                row_df[col] = np.nan

        # TARGET ENCODING COLUMNA x COLUMNA
        for col in categorical_high_card:
            enc = te_encoders[col]                # encoder correcto
            te_val = enc.transform(row_df[[col]]).values.flatten()
            row_df[f"te_{col}"] = te_val

        # Reordenar columnas según pipeline
        expected = numeric_features + categorical_low_card + te_cols

        return row_df.reindex(columns=expected, fill_value=np.nan)

    def predict_price_log(single_row_df, model_pipeline, corr_model=None):

        if isinstance(single_row_df, dict):
            single_row_df = pd.DataFrame([single_row_df])
        if isinstance(single_row_df, pd.Series):
            single_row_df = single_row_df.to_frame().T

        # Eliminar duplicados
        single_row_df = single_row_df.loc[:, ~single_row_df.columns.duplicated()]

        # Reordenar columnas según las del pipeline
        if hasattr(model_pipeline, "feature_names_in_"):
            expected_cols = list(model_pipeline.feature_names_in_)
        else:
            expected_cols = numeric_features + categorical_low_card + te_cols

        single_row_df = single_row_df.reindex(columns=expected_cols, fill_value=np.nan)

        pred_log = float(model_pipeline.predict(single_row_df)[0])

        if corr_model is not None:
            pred_log += float(corr_model.predict(np.array(pred_log).reshape(-1, 1))[0])

        return pred_log

    def calcular_precio_base_contextual(row, df_base):
        
        raza = row.get('raza')
        color = row.get('color')
        cat = row.get('raza_categoria')
        tienda = row.get('tienda')
        genero = row.get('genero')

        # raza + color + genero
        if raza and color and genero:
            sub = df_base[(df_base['raza'] == raza) & (df_base['color'] == color) & (df_base['genero'] == genero) & df_base['precio_venta'].notna()]
            if len(sub):
                return float(sub['precio_venta'].median())

        # raza sola
        sub = df_base[(df_base['raza'] == raza) & df_base['precio_venta'].notna()]
        if len(sub):
            return float(sub['precio_venta'].median())

        # categoría de raza
        if cat:
            sub = df_base[(df_base['raza_categoria'] == cat) & df_base['precio_venta'].notna()]
            if len(sub):
                return float(sub['precio_venta'].median())

        # tienda
        if tienda:
            sub = df_base[(df_base['tienda'] == tienda) & df_base['precio_venta'].notna()]
            if len(sub):
                return float(sub['precio_venta'].median())

        # global
        global_mean = df_base['precio_venta'].median()
        return float(global_mean if pd.notna(global_mean) else 0.0)


    def sugerir_precio_dinamico(perro_row, model_pipeline, te_full, clf_pipeline=None, corr_model=None, default_margin=DEFAULT_MARGIN):
        
        r = perro_row.to_dict() if isinstance(perro_row, pd.Series) else dict(perro_row)

        X_row = preparar_fila_inferencia(r, te_full, numeric_features, categorical_low_card, categorical_high_card, te_cols)
        
        pred_log = predict_price_log(X_row, model_pipeline, corr_model=corr_model)
        pred_price = float(np.expm1(pred_log))

        if pd.notna(r.get('base')):
            precio_hist = float(r['base'])
        else:
            precio_hist = calcular_precio_base_contextual(r, train_df)

        dias_val = r.get('dias_en_tienda', 0)
        dias = int(0 if pd.isna(dias_val) else dias_val)
        vistas_val = r.get('vistas_7d', 0)
        vistas = int(0 if pd.isna(vistas_val) else vistas_val)

        categoria = r.get('raza_categoria', None)
        raza = r.get('raza', None)

        descuento = 0.0
        if raza == 'dachshund' and dias < 28:
            descuento = descuento
        elif categoria in discount_rules:
            for d, desc in sorted(discount_rules[categoria]):
                if dias >= d:
                    descuento += desc
        else:
            descuento = descuento

        demanda_adj = 0#-0.05 if vistas < 10 else (0.10 if vistas >= 100 else (0.05 if vistas >= 50 else 0.0))

        precio_desc = pred_price * (1 - descuento)
        precio_sugerido = precio_desc * (1 + demanda_adj + default_margin)

        min_val = pd.to_numeric(r.get('min'), errors='coerce')
        if pd.notna(min_val):
            floor_price = min_val
        else:
            floor_price = precio_hist * DEFAULT_FLOOR_PCT

        ajuste_tipo = None
        if precio_sugerido < floor_price:
            precio_sugerido = floor_price
            ajuste_tipo = 'floor_aplicado'

        return {
            'precio_modelo': round(pred_price, 2),
            'precio_sugerido': round(precio_sugerido, 2),
            'applied_discount_pct': round(descuento, 4),
            'demand_adj_pct': round(demanda_adj, 4),
            'dias_en_tienda': dias,
            'categoria': categoria,
            'vistas_7d': vistas,
            'ajuste_aplicado': bool(ajuste_tipo),
            'tipo_ajuste': ajuste_tipo,
            'precio_base': precio_hist
        }

    def recomendar_reubicacion(perro_row, tiendas_list, model_pipeline, te_full, clf_pipeline=None, corr_model=None, top_k=3):

        resultados = []
        for t in tiendas_list:
            
            perro_row['tienda'] = t
            perro_row_df = perro_row.to_frame().T
            perro_row_df = perro_row_df.drop(columns=['precio_mean_tienda', 'prob_vender_tienda', 'raza_x_tienda'])
            perro_row_df = perro_row_df.merge(agg_t[['tienda', 'precio_mean_tienda', 'prob_vender_tienda']], on='tienda', how='left')
            perro_row_df['raza_x_tienda'] = perro_row_df['raza'].astype(str) + '___' + perro_row_df['tienda'].astype(str)
            perro_row = perro_row_df.iloc[0]
        
            X_row = preparar_fila_inferencia(perro_row, te_full, numeric_features, categorical_low_card, categorical_high_card, te_cols)
       
            try:
                p_log = predict_price_log(X_row, model_pipeline, corr_model=corr_model)
                p = float(np.expm1(p_log))
            except Exception:
                p = np.nan
            X_row['dias_en_tienda'] = perro_row.get('dias_en_tienda')
            
            prob = 1
            score = p * prob if (not pd.isna(p) and not pd.isna(prob)) else np.nan
            resultados.append({'tienda': t, 'precio_esperado': p, 'prob_venta': prob, 'score': score})
        return pd.DataFrame(resultados).sort_values('score', ascending=False).head(top_k)

    def aproximar_precio(precio_venta, precios_disponibles, precio_base):

        precios_disponibles = sorted(precios_disponibles)
        
        if precio_venta <= precios_disponibles[0]:
            return precios_disponibles[0]
        
        if precio_venta >= precios_disponibles[-1]:
            return precios_disponibles[-1]

        for i in range(len(precios_disponibles)-1):
            p_inf = precios_disponibles[i]
            p_sup = precios_disponibles[i+1]
            
            # Validamos si el precio_venta cae en el intervalo actual
            if p_inf <= precio_venta <= p_sup:
                # Calcular el descuento relativo al precio_base para el precio inferior
                descuento_inf = (precio_base - p_inf) / precio_base

                # Definir el umbral_relativo según el descuento
                if descuento_inf <= 0.10:
                    umbral_relativo = 0.05  # 5% de umbral si el descuento es 10% o menor
                elif descuento_inf <= 0.20:
                    umbral_relativo = 0.06  # 6% de umbral si el descuento es 20% o menor
                elif descuento_inf <= 0.25:
                    umbral_relativo = 0.08  # 8% de umbral si el descuento es 25% o menor
                elif descuento_inf <= 0.35:
                    umbral_relativo = 0.10  # 10% de umbral si el descuento es 35% o menor
                else:
                    umbral_relativo = 0.12  # 12% de umbral si el descuento es mayor a 35%

                # Condición para verificar si el precio_venta está cerca del precio inferior
                if (precio_venta - p_inf) / p_inf <= umbral_relativo:
                    return p_inf
                # Si el precio_venta no está cerca del precio inferior, usamos el precio superior
                else:
                    return p_sup
        
        # Si no se encuentra en ningún intervalo, se usa el precio más cercano
        return precios_disponibles[-1]  # O el precio más cercano según el caso


    # In[15]:


    # Exportar recomendaciones (no vendidos)

    #unv = predict_df[predict_df['last']==1].copy().reset_index(drop=True)
    #.loc[predict_df["id"].isin(["932002000719523","900255002110908"])].head(2)
    unv = predict_df.copy().reset_index(drop=True)
    unv = unv.loc[:, ~unv.columns.duplicated()].copy()

    tiendas_list = df['tienda'].unique().tolist()
    rows = []
    # Limpieza preventiva: corregir categorías duplicadas en OHE
    ohe = best_model.named_steps['preproc'].named_transformers_['catlow'].named_steps['ohe']
    if hasattr(ohe, 'categories_'):
        for i, cats in enumerate(ohe.categories_):
            ohe.categories_[i] = np.unique(cats)

    def safe_int_convert(value):
        if pd.notna(value):
            return int(float(value))
        else:
            return np.nan

    for _, perro in unv.iterrows():
        tienda_orig = ''
        
        tienda_orig = perro['tienda']
        reloc = recomendar_reubicacion(perro, tiendas_list, best_model, TE_ENCODERS, corr_model=corr_model, top_k=1)
     
        if not reloc.empty:
            tienda_recom = reloc.iloc[0]['tienda']
            prob_recom = float(reloc.iloc[0]['prob_venta']) if not pd.isna(reloc.iloc[0]['prob_venta']) else np.nan
        else:
            tienda_recom = None
            prob_recom = np.nan
       
        perro['tienda'] = tienda_recom
        
        if isinstance(perro, pd.Series):
            perro = perro.to_frame().T
        
        perro = perro.drop(['precio_mean_tienda','prob_vender_tienda','raza_x_tienda'], axis=1)

        perro = perro.merge(
            agg_t[['tienda','precio_mean_tienda','prob_vender_tienda']],
            on='tienda',
            how='left'
        )
        
        perro['raza_x_tienda'] = perro['raza'].astype(str) + '___' + perro['tienda'].astype(str)
        perro = perro.iloc[0]
      
        sug = sugerir_precio_dinamico(perro, best_model, TE_ENCODERS, corr_model=corr_model)

        valores = [desc for (_, desc) in discount_rules[perro.get('raza_categoria')]]
        base = float(perro.get('base'))
        acumulado = 0
        precios_disponibles = []
        for v in valores:
            acumulado += v
            precios_disponibles.append(base * (1 - acumulado))
        precio_aprox = aproximar_precio(sug['precio_sugerido'], precios_disponibles, base)

        rows.append({
            'id': perro.get('id'),
            'raza': perro.get('raza'),
            'color': perro.get('color'),
            'genero': perro.get('genero'),
            'tamano': perro.get('tamano'),
            'caracteristica': perro.get('caracteristica'),
            'registro': perro.get('registro'),
            'precio_base': safe_int_convert(sug['precio_base']) if pd.notna(sug['precio_base']) else np.nan,
            'precio_venta_hist': safe_int_convert(perro.get('precio_venta', np.nan)) if pd.notna(perro.get('precio_venta')) else np.nan,
            'precio_modelo': safe_int_convert(sug['precio_modelo']) if pd.notna(sug['precio_modelo']) else np.nan,
            'precio_sugerido': safe_int_convert(sug['precio_sugerido']) if pd.notna(sug['precio_sugerido']) else np.nan,
            'precio_aproximado': safe_int_convert(precio_aprox) if pd.notna(precio_aprox) else np.nan,
            'applied_discount_pct': sug['applied_discount_pct'] * -1,
            'demand_adj_pct': sug['demand_adj_pct'],
            'tienda_actual': tienda_orig,
            'tienda_recomendada': tienda_recom,
            'raza_categoria': perro.get('raza_categoria'),
            'dias_en_tienda': perro.get('dias_en_tienda'),
            'vistas_7d': perro.get('vistas_7d'),
            'fecha_generacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'ajuste_aplicado': sug['ajuste_aplicado'],
            'tipo_ajuste': sug['tipo_ajuste'],
            'comentario': perro.get('comentario')
        })

    recs_df = pd.DataFrame(rows)
    numeric_cols = ['precio_modelo', 'precio_sugerido', 'applied_discount_pct', 'demand_adj_pct', 'precio_venta_hist']
    recs_df[numeric_cols] = recs_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    worksheet_ejecucion_recomendaciones = sheet_ejecucion.add_worksheet(title=f'recomendaciones_{fecha_str}', rows="10000", cols="50")
    recs_df_send = recs_df.fillna("").copy()
    datos = recs_df_send.values.tolist()
    encabezados = recs_df_send.columns.tolist()
    worksheet_ejecucion_recomendaciones.update('A1', [encabezados] + datos)


    # In[16]:


    # Actualizar Google Sheet

    def safe_float(x):
        try:
            x_str = str(x).strip()
            if x_str == "":
                return np.nan
            return float(x_str)
        except:
            return np.nan

    def safe_int_convert(x):
        try:
            if pd.isna(x):
                return ""
            return int(float(x))
        except:
            return ""
            
    def actualizar_sheet_incremental_batch(worksheet, gs_df, recs_df):

        header = worksheet.row_values(1)
        header_map = {c: i + 1 for i, c in enumerate(header)}

        if "Price" not in header:
            raise ValueError("❌ La hoja no tiene columna 'Price'")

        price_main_col = header_map["Price"]

        updates = []
        new_columns = []

        for _, row in recs_df.iterrows():
            perro_id = str(row["id"]).strip().lower()
            nuevo_precio = safe_float(row["precio_aproximado"])
            fecha_actual = row["fecha_generacion"]

            if np.isnan(nuevo_precio):
                continue

            match = gs_df[
                gs_df["Microchip #"]
                .astype(str)
                .str.lower()
                .str.strip()
                == perro_id
            ]

            if match.empty:
                continue

            idx = match.index[0]
            sheet_row = idx + 2  # header ocupa fila 1

            # SIEMPRE actualizar Price (precio vigente)
            updates.append(
                gspread.Cell(
                    row=sheet_row,
                    col=price_main_col,
                    value=safe_int_convert(nuevo_precio)
                )
            )

            # HISTÓRICO SOLO SI CAMBIA
            price_cols = sorted(
                [c for c in gs_df.columns if c.startswith("Price_")],
                key=lambda x: int(x.split("_")[1])
            )

            last_price_val = np.nan
            for col in reversed(price_cols):
                val = safe_float(gs_df.loc[idx, col])
                if not np.isnan(val):
                    last_price_val = val
                    break

            # Si es el mismo precio → NO crear histórico
            if not np.isnan(last_price_val) and last_price_val == nuevo_precio:
                continue

            # Buscar slot libre o crear siguiente
            next_n = 1
            for col in price_cols:
                if pd.isna(gs_df.loc[idx, col]) or str(gs_df.loc[idx, col]).strip() == "":
                    next_n = int(col.split("_")[1])
                    break
            else:
                if price_cols:
                    next_n = int(price_cols[-1].split("_")[1]) + 1

            price_col = f"Price_{next_n:02d}"
            date_col  = f"Date_{next_n:02d}"

            # Crear columnas si no existen
            for col in (price_col, date_col):
                if col not in header:
                    new_columns.append(col)
                    header.append(col)
                    header_map[col] = len(header)

            # Registrar histórico
            updates.append(
                gspread.Cell(
                    row=sheet_row,
                    col=header_map[price_col],
                    value=safe_int_convert(nuevo_precio)
                )
            )

            updates.append(
                gspread.Cell(
                    row=sheet_row,
                    col=header_map[date_col],
                    value=datetime.strptime(
                        fecha_actual, "%Y-%m-%d %H:%M:%S"
                    ).strftime("%d-%m-%Y")
                )
            )

        # Crear columnas nuevas una sola vez
        if new_columns:
            worksheet.add_cols(len(new_columns))
            worksheet.update('1:1', [header])

        # Batch update
        if updates:
            worksheet.update_cells(updates)
            print(f"✔ Actualización completada ({len(updates)} celdas)")
        else:
            print("ℹ No hubo actualizaciones")

    def actualizar_sheet_price_batch(worksheet, gs_df, recs_df):

        header = worksheet.row_values(1)
        header_map = {c: i + 1 for i, c in enumerate(header)}

        if "Microchip#" not in header:
            raise ValueError("❌ La hoja no tiene columna 'Microchip #'")

        if "Price" not in header:
            raise ValueError("❌ La hoja no tiene columna 'Price'")

        price_col_idx = header_map["Price"]

        date_col_idx = header_map.get("Date", None)

        updates = []

        for _, row in recs_df.iterrows():
            perro_id = str(row["id"]).strip().lower()
            nuevo_precio = safe_float(row["precio_aproximado"])

            if np.isnan(nuevo_precio):
                continue

            match = gs_df[
                gs_df["Microchip#"]
                .astype(str)
                .str.strip()
                .str.lower()
                == perro_id
            ]

            if match.empty:
                continue

            idx = match.index[0]
            sheet_row = idx + 2

            precio_actual = safe_float(gs_df.loc[idx, "Price"])

            if not np.isnan(precio_actual) and precio_actual == nuevo_precio:
                continue

            updates.append(
                gspread.Cell(
                    row=sheet_row,
                    col=price_col_idx,
                    value=safe_int_convert(nuevo_precio)
                )
            )

            if date_col_idx:
                updates.append(
                    gspread.Cell(
                        row=sheet_row,
                        col=date_col_idx,
                        value=datetime.now().strftime("%d-%m-%Y")
                    )
                )

        # Ejecutar batch update
        if updates:
            worksheet.update_cells(updates)
            print(f"✔ {len(updates)} celdas actualizadas")
        else:
            print("ℹ No hubo cambios de precio")

    if NEW_RECORDS:
        actualizar_sheet_price_batch(worksheet_new, df_sheet_new, recs_df)
    else:
        actualizar_sheet_incremental_batch(worksheet, df_sheet, recs_df)


    # In[17]:


    '''def pipeline_completo:
        ...

    try:
        
        print("✅ Proceso ejecutado correctamente")

        enviar_correo(
            asunto="✅ Modelo Mascotas — Ejecución Exitosa",
            mensaje=f"""
    El modelo se ejecutó correctamente.

    Fecha: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}
    Estado: OK
            """
        )

    except Exception as e:
        error_trace = traceback.format_exc()

        enviar_correo(
            asunto="❌ Modelo Mascotas — ERROR en Ejecución",
            mensaje=f"""
    El modelo falló durante la ejecución.

    Fecha: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}

    Error:
    {str(e)}

    Traceback:
    {error_trace}
            """
        )
        raise  # importante para que Task Scheduler / Airflow marque error'''
        
    '''
    enviar_correo(
        "🧪 Test correo automático",
        "Si ves esto, el sistema de notificaciones funciona."
    )'''


    # In[18]:


    spreadsheet = worksheet.spreadsheet

    total_cells = 0
    for ws in spreadsheet.worksheets():
        cells = ws.row_count * ws.col_count
        print(ws.title, cells)
        total_cells += cells

    print("TOTAL:", total_cells)


    # In[19]:


    rows = worksheet.row_count
    cols = worksheet.col_count
    print(rows * cols)


    # In[20]:


    # Guardar artefactos y metricas metricas

    if TRAIN:
        metrics = {
            'mae_before_correction': float(mean_absolute_error(y_true, y_pred)),
            'mae_after_correction': float(mean_absolute_error(y_true, y_pred_corr_orig)),
            'rmse': float(rmse),
            'r2': float(r2)
        }

        joblib.dump(
            {
                'best_model': best_model,
                'corr_model': corr_model,
                'metrics': metrics,
                'final_feature_names': final_feature_names
            },
            ARTIFACTS_MODEL_PATH
        )

        print("✔ Artefactos guardados en:", ARTIFACTS_MODEL_PATH)
        print("Metrics:", metrics)


    # In[21]:


    # Evaluación, Importancia de Características y Nombres Reales

    if TRAIN:
        # Determinar el mejor modelo del GridSearch o RandomizedSearch
        search_obj = gs if 'gs' in locals() else rs
        best_model = search_obj.best_estimator_
        
        # EVALUACIÓN DEL MODELO
        # Asegurarse de que X_test tiene todas las columnas target-encoded
        X_test = df.loc[X_test.index, numeric_features + categorical_low_card + te_cols]
        
        # Predicciones en escala logarítmica (ya que el modelo fue entrenado en log)
        y_pred_log = best_model.predict(X_test)
        
        # Métricas en log
        rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))  # Raíz cuadrada de error cuadrático
        mae_log = np.mean(np.abs(y_test - y_pred_log))  # Error absoluto medio
        r2_log = r2_score(y_test, y_pred_log)  # R² en escala logarítmica
        
        # Métricas en escala real (deshacer la transformación logarítmica)
        y_pred_real = np.expm1(y_pred_log)  # Recuperar los valores en la escala real
        y_true_real = np.expm1(y_test)  # También para los valores reales
        
        rmse_real = np.sqrt(mean_squared_error(y_true_real, y_pred_real))  # RMSE en escala real
        mae_real = np.mean(np.abs(y_true_real - y_pred_real))  # MAE en escala real
        r2_real = r2_score(y_true_real, y_pred_real)  # R² en escala real
        
        # Imprimir resultados
        print(f"Resultados del Modelo Optimizado:")
        print(f"RMSE (log)  = {rmse_log:.4f}")
        print(f"MAE (log)   = {mae_log:.4f}")
        print(f"R² (log)    = {r2_log:.3f}")
        print(f"RMSE (real) = {rmse_real:.2f}")
        print(f"MAE (real)  = {mae_real:.2f}")
        print(f"R² (real)   = {r2_real:.3f}")
        
        # GRAFICOS DE DIAGNÓSTICO
        # --- 1. Real vs Predicho ---
        plt.figure(figsize=(6,6))
        sns.scatterplot(x=y_true_real, y=y_pred_real, alpha=0.6)
        plt.plot([y_true_real.min(), y_true_real.max()], [y_true_real.min(), y_true_real.max()], 'r--', label='Ideal')
        plt.xlabel("Valor Real")
        plt.ylabel("Predicción Modelo")
        plt.title("Predicción vs Valor Real")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # --- 2. Distribución de errores ---
        residuals = y_true_real - y_pred_real
        plt.figure(figsize=(8,5))
        sns.histplot(residuals, bins=30, kde=True)
        plt.title("Distribución de Errores (Residuos)")
        plt.xlabel("Error (Valor Real - Predicho)")
        plt.ylabel("Frecuencia")
        plt.tight_layout()
        plt.show()
        
        # IMPORTANCIA DE VARIABLES (LightGBM)
        try:
            # Acceder al modelo LightGBM dentro del Pipeline
            booster = best_model.named_steps['model'].booster_
            importance = booster.feature_importance(importance_type='gain')
            feature_names = booster.feature_name()
        
            # Si los nombres de las características son genéricos (ej. Column_0, Column_1...), reconstruir los nombres reales
            if feature_names == ['Column_' + str(i) for i in range(len(feature_names))]:
                # Reconstruir los nombres de las características si hay un preprocesador
                if 'preproc' in best_model.named_steps:
                    preprocessor = best_model.named_steps['preproc']
                    num_names = numeric_features  # Asumiendo que ya tienes la lista de features numéricas
        
                    # Obtener los nombres de las columnas codificadas
                    ohe_transformer = preprocessor.named_transformers_['catlow']['ohe']
                    ohe_cols = list(ohe_transformer.get_feature_names_out(categorical_low_card))
                    high_card_names = categorical_high_card  # Si tienes columnas de alta cardinalidad
                    final_feature_names = num_names + ohe_cols + high_card_names
        
                    # Alinear las longitudes de importancia y nombres
                    min_len = min(len(importance), len(final_feature_names))
                    importance = importance[:min_len]
                    final_feature_names = final_feature_names[:min_len]
                    fi_df = pd.DataFrame({'feature': final_feature_names, 'importance': importance})
                else:
                    # Si no hay preprocesador, usar los nombres originales (Column_0, Column_1, etc.)
                    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
            else:
                fi_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
        
            # Ordenar las importancias y mostrar las top 15
            fi_df = fi_df.sort_values(by='importance', ascending=False)
        
            print("\nTop 15 características más importantes:")
            display(fi_df.head(15))
        
            # Gráfico de importancia de características
            plt.figure(figsize=(10,6))
            sns.barplot(data=fi_df.head(20), x='importance', y='feature', palette='viridis')
            plt.title("Importancia de Características (LightGBM)")
            plt.xlabel("Ganancia acumulada")
            plt.ylabel("Feature")
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print("\nNo se pudo extraer la importancia de características.")
            print("Error:", e)


    # In[22]:


    if TRAIN:
        # SHAP + Proxy Scoring
        # 1) Extraer modelo y preprocesador
        model_lgb = best_model.named_steps['model']
        preprocessor = best_model.named_steps['preproc']
        
        # 2) Transformar X_test usando el preprocesador exactamente como se entrenó
        #    OJO: solo las columnas de entrada originales (sin duplicar te_cols)
        X_test_input = X_test[numeric_features + categorical_low_card + te_cols]
        X_test_transformed = preprocessor.transform(X_test_input)
        
        # Reconstruir nombres de features finales tras preprocesamiento
        ohe_cols = list(preprocessor.named_transformers_['catlow']
                        .named_steps['ohe']
                        .get_feature_names_out(categorical_low_card))
        final_feature_names = numeric_features + ohe_cols + te_cols
        
        # 3) Crear TreeExplainer
        explainer = shap.TreeExplainer(model_lgb)
        shap_values = explainer.shap_values(X_test_transformed)
        
        # 4) Summary plot
        shap.summary_plot(
            shap_values,
            pd.DataFrame(X_test_transformed, columns=final_feature_names),
            max_display=25
        )
        
        # 5) Analizar interacciones
        shap_interact = explainer.shap_interaction_values(X_test_transformed)
        mean_interact = np.abs(shap_interact).mean(axis=0).sum(axis=1)
        interact_df = pd.DataFrame({
            'feature': final_feature_names,
            'mean_interaction': mean_interact
        }).sort_values('mean_interaction', ascending=False)
        
        print("\nTop features por interacción media:")
        display(interact_df.head(20))
        
        # Ejemplo de dependencias:
        # shap.dependence_plot('te_raza', shap_values, pd.DataFrame(X_test_transformed, columns=final_feature_names))
        # shap.dependence_plot('te_color', shap_values, pd.DataFrame(X_test_transformed, columns=final_feature_names))
        
        # 6) Proxy Scoring Function
        def feature_proxy_score(df_full, feature_name, target_col='precio_venta', time_col='fecha_listado', n_splits=3):
            """
            Calcula un proxy-score combinando:
              - correlación Spearman con el target
              - importancia SHAP (aprox. vía permutation importance)
              - estabilidad temporal
            """
            # 1) Correlación Spearman
            if feature_name in df_full.columns and pd.api.types.is_numeric_dtype(df_full[feature_name]):
                corr = abs(df_full[[feature_name, target_col]].dropna().corr(method='spearman').iloc[0, 1])
            else:
                corr = 0.0
        
            # 2) SHAP/proxy importance vía permutation importance
            try:
                perm = permutation_importance(
                    best_model,
                    X_test[numeric_features + categorical_low_card],
                    y_test,
                    n_repeats=5,
                    random_state=RND,
                    n_jobs=-1
                )
                idx = final_feature_names.index(feature_name) if feature_name in final_feature_names else None
                shap_imp = perm.importances_mean[idx] if idx is not None else 0.0
            except Exception:
                shap_imp = 0.0
        
            # 3) Estabilidad temporal
            stab_change = 0.0
            if time_col in df_full.columns:
                df_full = df_full.dropna(subset=[time_col])
                df_full['period'] = pd.qcut(
                    df_full[time_col].view('int64') // 10**9,  # convierte a segundos desde epoch
                    q=n_splits, 
                    duplicates='drop'
                )
        
                imps = []
                for p in df_full['period'].unique():
                    sub = df_full[df_full['period'] == p]
                    if len(sub) < 50:
                        continue
                    try:
                        Xsub = sub[numeric_features + categorical_low_card]
                        imp_sub = permutation_importance(best_model, Xsub, sub['precio_venta'], n_repeats=3, random_state=RND, n_jobs=1)
                        idx = final_feature_names.index(feature_name) if feature_name in final_feature_names else None
                        if idx is not None:
                            imps.append(imp_sub.importances_mean[idx])
                    except Exception:
                        pass
                if len(imps) >= 2:
                    stab_change = float(np.std(imps))
        
            # 4) Score combinado
            score = 0.5 * shap_imp + 0.3 * corr - 0.2 * stab_change
            return {'feature': feature_name, 'corr': corr, 'shap_imp': shap_imp, 'stab': stab_change, 'proxy_score': score}
        
        # 7) Calcular proxy scores
        proxy_scores = [feature_proxy_score(df, f) for f in final_feature_names if f.startswith('te_')]
        proxy_scores_df = pd.DataFrame(proxy_scores).sort_values('proxy_score', ascending=False)
        
        print("\nProxy scores (top):")
        display(proxy_scores_df.head(20))

    pass

if __name__ == "__main__":
    main()
