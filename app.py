import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import plotly.graph_objects as go
import os
import unicodedata

st.set_page_config(page_title="Fantacalcio Pro", layout="wide")
st.title("⚽ Fantacalcio: Motore Inferenziale Avanzato")

# --- 1. MATRICE DIFENSIVA IBRIDA (Dinamica + Statica) ---
DIFFICOLTA_GRANULARE_DEFAULT = {
    'Inter': {'P': 5, 'D': 5, 'C': 3, 'A': 3}, 'Lazio': {'P': 5, 'D': 5, 'C': 5, 'A': 5},
    'Roma': {'P': 5, 'D': 5, 'C': 5, 'A': 5}, 'Milan': {'P': 5, 'D': 5, 'C': 5, 'A': 5},
    'Como': {'P': 4, 'D': 4, 'C': 5, 'A': 5}, 'Frosinone': {'P': 5, 'D': 5, 'C': 3, 'A': 3},
    'Juventus': {'P': 3, 'D': 3, 'C': 4, 'A': 4}, 'Torino': {'P': 3, 'D': 3, 'C': 3, 'A': 3},
    'Cagliari': {'P': 1, 'D': 1, 'C': 4, 'A': 4}, 'Napoli': {'P': 3, 'D': 3, 'C': 3, 'A': 3},
    'Atalanta': {'P': 4, 'D': 4, 'C': 2, 'A': 2}, 'Lecce': {'P': 2, 'D': 2, 'C': 2, 'A': 2},
    'Bologna': {'P': 2, 'D': 2, 'C': 3, 'A': 3}, 'Sassuolo': {'P': 3, 'D': 3, 'C': 2, 'A': 2},
    'Monza': {'P': 4, 'D': 4, 'C': 1, 'A': 1}, 'Genoa': {'P': 1, 'D': 1, 'C': 3, 'A': 3},
    'Udinese': {'P': 2, 'D': 2, 'C': 1, 'A': 1}, 'Fiorentina': {'P': 2, 'D': 2, 'C': 1, 'A': 1},
    'Parma': {'P': 1, 'D': 1, 'C': 2, 'A': 2}, 'Venezia': {'P': 1, 'D': 1, 'C': 1, 'A': 1}
}

def carica_difficolta():
    if os.path.exists('classifica.csv'):
        try:
            df_classifica = pd.read_csv('classifica.csv', sep=';')
            soglie_gf = df_classifica['GF'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
            soglie_gs = df_classifica['GS'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()

            def assegna_difficolta(valore, soglie, inverti=False):
                if not inverti:
                    if valore <= soglie[0]: return 1
                    if valore <= soglie[1]: return 2
                    if valore <= soglie[2]: return 3
                    if valore <= soglie[3]: return 4
                    return 5
                else:
                    if valore <= soglie[0]: return 5
                    if valore <= soglie[1]: return 4
                    if valore <= soglie[2]: return 3
                    if valore <= soglie[3]: return 2
                    return 1

            diff_dinamica = {}
            for _, row in df_classifica.iterrows():
                sq = str(row['Squadra']).strip()
                diff_pd = assegna_difficolta(row['GF'], soglie_gf, inverti=False)
                diff_ca = assegna_difficolta(row['GS'], soglie_gs, inverti=True)
                diff_dinamica[sq] = {'P': diff_pd, 'D': diff_pd, 'C': diff_ca, 'A': diff_ca}
            return diff_dinamica
        except Exception:
            return DIFFICOLTA_GRANULARE_DEFAULT
    return DIFFICOLTA_GRANULARE_DEFAULT

DIFFICOLTA_GRANULARE = carica_difficolta()

MODULI_VALIDI = [
    {'nome': '3-4-3', 'P': 1, 'D': 3, 'C': 4, 'A': 3}, {'nome': '3-5-2', 'P': 1, 'D': 3, 'C': 5, 'A': 2},
    {'nome': '4-3-3', 'P': 1, 'D': 4, 'C': 3, 'A': 3}, {'nome': '4-4-2', 'P': 1, 'D': 4, 'C': 4, 'A': 2},
    {'nome': '4-5-1', 'P': 1, 'D': 4, 'C': 5, 'A': 1}, {'nome': '5-3-2', 'P': 1, 'D': 5, 'C': 3, 'A': 2},
    {'nome': '5-4-1', 'P': 1, 'D': 5, 'C': 4, 'A': 1}
]
BASELINE_RUOLO = {'P': 5.0, 'D': 5.5, 'C': 6.0, 'A': 6.5}

def pulisci_testo(testo):
    if not isinstance(testo, str): return ""
    return unicodedata.normalize('NFKD', testo).encode('ASCII', 'ignore').decode('utf-8').lower().strip()

# --- 2. ZONA CARICAMENTO DATI (SIDEBAR) ---
st.sidebar.header("⚙️ Configurazione Giornata")
giornata_input = st.sidebar.number_input("Imposta Giornata", min_value=1, max_value=38, value=3)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Dati Giocatori (FBref)")
file_fbref = st.sidebar.file_uploader("Trascina statistiche giocatori", type=['csv', 'txt', 'xls'])

if file_fbref is not None:
    with st.spinner("Analisi inferenziale del file in corso..."):
        try:
            df_utili = None
            for separatore in [',', ';']:
                for enc in ['utf-8', 'latin-1']:
                    for skip in [0, 1]:
                        try:
                            file_fbref.seek(0)
                            temp = pd.read_csv(file_fbref, sep=separatore, skiprows=skip, on_bad_lines='skip', encoding=enc)
                            if 'Player' in temp.columns: temp = temp.rename(columns={'Player': 'Giocatore'})
                            if 'Giocatore' in temp.columns:
                                df_utili = temp
                                break
                        except Exception:
                            continue
                    if df_utili is not None: break
                if df_utili is not None: break
            
            if df_utili is not None and 'Giocatore' in df_utili.columns:
                xg_col = 'xG' if 'xG' in df_utili.columns else None
                xag_col = 'xAG' if 'xAG' in df_utili.columns else ('xA' if 'xA' in df_utili.columns else None)
                
                if not xg_col and 'Gls' in df_utili.columns:
                    df_utili['xG_stimato'] = df_utili['Gls'].astype(str).str.replace(',', '.')
                    xg_col = 'xG_stimato'
                if not xag_col and 'Ast' in df_utili.columns:
                    df_utili['xA_stimato'] = df_utili['Ast'].astype(str).str.replace(',', '.')
                    xag_col = 'xA_stimato'

                if xg_col and xag_col:
                    df_utili = df_utili[['Giocatore', xg_col, xag_col]].copy()
                    df_utili = df_utili[df_utili['Giocatore'] != 'Giocatore']
                    df_utili['xG'] = pd.to_numeric(df_utili[xg_col], errors='coerce').fillna(0.0)
                    df_utili['xA'] = pd.to_numeric(df_utili[xag_col], errors='coerce').fillna(0.0)
                    
                    if os.path.exists('giocatori.csv'):
                        mio_file = pd.read_csv('giocatori.csv', sep=';').fillna(0)
                        df_utili['Giocatore_Clean'] = df_utili['Giocatore'].apply(pulisci_testo)
                        
                        def trova_xg_xa(nome_giocatore):
                            nome_clean = pulisci_testo(nome_giocatore)
                            if not nome_clean: return 0.0, 0.0
                            # Priorità a match esatto, altrimenti parziale
                            exact = df_utili[df_utili['Giocatore_Clean'] == nome_clean]
                            if not exact.empty: return exact['xG'].values[0], exact['xA'].values[0]
                            match = df_utili[df_utili['Giocatore_Clean'].str.contains(nome_clean, case=False, na=False)]
                            if not match.empty: return match['xG'].values[0], match['xA'].values[0]
                            return 0.0, 0.0
                        
                        risultati = mio_file['Nome'].apply(trova_xg_xa)
                        mio_file['xG'] = [res[0] for res in risultati]
                        mio_file['xA'] = [res[1] for res in risultati]
                        mio_file.to_csv('giocatori.csv', sep=';', index=False)
                        st.cache_data.clear()
                        st.sidebar.success("✅ Dati offensivi aggiornati!")
                    else:
                        st.sidebar.error("❌ Manca il file 'giocatori.csv'.")
                else:
                    st.sidebar.error("❌ File privo di metriche xG/xA o Gol/Assist.")
            else:
                st.sidebar.error("❌ Colonna 'Giocatore' o 'Player' non trovata.")
        except Exception as e:
            st.sidebar.error(f"Errore tecnico: {e}")

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Dati Squadre (FBref)")
file_classifica_upload = st.sidebar.file_uploader("Trascina statistiche Squadre", type=['csv', 'xls', 'txt'])

if file_classifica_upload is not None:
    try:
        df_c = None
        if file_classifica_upload.name.lower().endswith(('.xls', '.html')):
            try:
                file_classifica_upload.seek(0)
                dfs = pd.read_html(file_classifica_upload)
                if dfs:
                    df_c = dfs[0]
                    if isinstance(df_c.columns, pd.MultiIndex):
                        df_c.columns = df_c.columns.get_level_values(-1)
            except Exception:
                pass
                
        if df_c is None:
            for sep in [',', ';']:
                for enc in ['utf-8', 'latin-1']:
                    for skip in [0, 1]:
                        try:
                            file_classifica_upload.seek(0)
                            temp_df = pd.read_csv(file_classifica_upload, sep=sep, skiprows=skip, on_bad_lines='skip', encoding=enc)
                            if 'Squad' in temp_df.columns or 'Squadra' in temp_df.columns:
                                df_c = temp_df
                                break
                        except Exception:
                            continue
                    if df_c is not None: break
                if df_c is not None: break
                
        if df_c is not None:
            col_squad = 'Squad' if 'Squad' in df_c.columns else 'Squadra'
            col_gf = 'Gls' if 'Gls' in df_c.columns else ('GF' if 'GF' in df_c.columns else None)
            col_gs = 'GA' if 'GA' in df_c.columns else ('GS' if 'GS' in df_c.columns else None)
            
            if col_gf and col_gs:
                df_clean = pd.DataFrame()
                df_clean['Squadra'] = df_c[col_squad].astype(str).str.replace(r'^[a-z]{2}\s+', '', regex=True).str.strip()
                df_clean['GF'] = pd.to_numeric(df_c[col_gf], errors='coerce').fillna(0)
                df_clean['GS'] = pd.to_numeric(df_c[col_gs], errors='coerce').fillna(0)
                
                df_clean = df_clean[df_clean['Squadra'] != 'nan']
                df_clean.to_csv('classifica.csv', sep=';', index=False)
                DIFFICOLTA_GRANULARE = carica_difficolta()
                st.cache_data.clear()
                st.sidebar.success("✅ Difficoltà Dinamiche ricalcolate e applicate!")
            else:
                st.sidebar.error("❌ Il file non contiene le colonne Gls/GF e GA/GS.")
        else:
            st.sidebar.error("❌ Colonna Squad/Squadra non trovata.")
    except Exception as e: 
        st.sidebar.error(f"Errore tecnico: {e}")

# --- 3. MODELLI INFERENZIALI E PREPARAZIONE DATI ---
@st.cache_resource
def allena_modelli_specializzati():
    dati_train = pd.DataFrame({
        'Voto_Base': [6.0, 5.0, 6.5, 6.0, 6.5, 5.5, 6.5, 7.0, 5.0, 6.0, 5.5, 6.5, 4.5, 6.0, 5.5, 6.5, 7.0, 4.0, 5.0, 7.5],
        'Media_Voto_Totale': [6.2, 5.5, 6.5, 6.0, 6.5, 5.8, 6.2, 6.8, 5.5, 6.2, 5.8, 6.5, 5.0, 6.0, 5.5, 6.2, 7.0, 4.5, 5.2, 7.1],
        'Bonus': [0.0, 0.0, 3.0, 0.5, 0.0, 0.0, 1.0, 3.0, -0.5, 0.0, -0.5, 1.0, -1.0, 0.0, -1.0, 0.0, 3.5, -2.0, -0.5, 4.0],
        'Fatica': [90, 90, 45, 0, 90, 75, 90, 90, 90, 90, 0, 45, 90, 90, 90, 90, 90, 90, 45, 90],
        'xG': [0.1, 0.0, 0.8, 0.3, 0.0, 0.1, 0.2, 0.5, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 1.2],
        'xA': [0.0, 0.0, 0.1, 0.2, 0.1, 0.0, 0.4, 0.3, 0.0, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.3],
        'Rigori': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        'Punizioni': [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        'Difficolta_Ruolo': [4, 5, 2, 3, 5, 3, 2, 1, 5, 3, 4, 2, 5, 3, 4, 2, 1, 5, 4, 1],
        'Fattore_Campo': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        'Ruolo': ['A', 'A', 'A', 'A', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'P', 'P', 'P', 'P', 'A', 'P', 'D', 'A']
    })
    
    form_base = 'Voto_Base ~ Media_Voto_Totale + Fatica + Difficolta_Ruolo * Fattore_Campo'
    form_bonus = 'Bonus ~ I(xG * 3 + xA * 1) + Rigori + Punizioni + Difficolta_Ruolo * Fattore_Campo'
    
    train_def = dati_train[dati_train['Ruolo'].isin(['P', 'D'])].copy()
    mod_base_def = smf.rlm(formula=form_base, data=train_def).fit()
    mod_bonus_def = smf.rlm(formula=form_bonus, data=train_def).fit()
    
    train_att = dati_train[dati_train['Ruolo'].isin(['C', 'A'])].copy()
    mod_base_att = smf.ols(formula=form_base, data=train_att).fit()
    mod_bonus_att = smf.ols(formula=form_bonus, data=train_att).fit()
    
    return {'def_base': mod_base_def, 'def_bonus': mod_bonus_def, 'att_base': mod_base_att, 'att_bonus': mod_bonus_att}

@st.cache_data
def prepara_dati_giornata(giornata_da_giocare):
    file_richiesti = ['giocatori.csv', 'calendario.csv', 'storico_voti_base.csv', 'storico_bonus.csv']
    mancanti = [f for f in file_richiesti if not os.path.exists(f)]
    if mancanti:
        return pd.DataFrame(), mancanti
            
    giocatori = pd.read_csv('giocatori.csv', sep=';').fillna(0)
    calendario = pd.read_csv('calendario.csv', sep=';')
    v_base = pd.read_csv('storico_voti_base.csv', sep=';').melt(id_vars=['Nome'], var_name='Giornata', value_name='Voto_Base')
    v_bonus = pd.read_csv('storico_bonus.csv', sep=';').melt(id_vars=['Nome'], var_name='Giornata', value_name='Bonus')
    
    v_base['Giornata'] = v_base['Giornata'].astype(str).str.replace('G', '', case=False).astype(int)
    v_bonus['Giornata'] = v_bonus['Giornata'].astype(str).str.replace('G', '', case=False).astype(int)
    
    storico = pd.merge(v_base, v_bonus, on=['Nome', 'Giornata'], how='outer')
    storico['Voto_Base'] = pd.to_numeric(storico['Voto_Base'], errors='coerce')
    storico['Bonus'] = pd.to_numeric(storico['Bonus'], errors='coerce').fillna(0.0)
    
    # Filtra solo le giornate precedenti a quella da giocare
    storico_pregresso = storico[storico['Giornata'] < giornata_da_giocare].dropna(subset=['Voto_Base'])
    df_completo = pd.merge(storico_pregresso, giocatori, on='Nome', how='left').sort_values(by=['Nome', 'Giornata'])
    
    # Calcolo EWM corretto (include anche l'ultima giornata disputata senza scartarla con shift)
    if not df_completo.empty:
        df_completo['Media_Voto_Totale'] = df_completo.groupby('Nome')['Voto_Base'].transform(
            lambda x: x.ewm(span=5, min_periods=1).mean()
        )
    
    partite_oggi = calendario[calendario['Giornata'] == giornata_da_giocare]
    prossima_giornata = []

    for _, row in giocatori.iterrows():
        nome = str(row['Nome']).strip()
        squadra = str(row['Squadra']).strip()
        ruolo = str(row['Ruolo']).strip().upper()
        
        storico_g = df_completo[df_completo['Nome'] == nome] if not df_completo.empty else pd.DataFrame()
        ultima_media = storico_g['Media_Voto_Totale'].iloc[-1] if not storico_g.empty else 5.5
        
        match_casa = partite_oggi[partite_oggi['Squadra_Casa'].str.strip().str.lower() == squadra.lower()]
        match_trasf = partite_oggi[partite_oggi['Squadra_Trasferta'].str.strip().str.lower() == squadra.lower()]
        
        if not match_casa.empty:
            fattore_campo = 1
            avversario = match_casa['Squadra_Trasferta'].values[0].strip()
        elif not match_trasf.empty:
            fattore_campo = 0
            avversario = match_trasf['Squadra_Casa'].values[0].strip()
        else:
            continue
            
        difficolta_ruolo = DIFFICOLTA_GRANULARE.get(avversario, {'P': 3, 'D': 3, 'C': 3, 'A': 3}).get(ruolo, 3)
        
        prossima_giornata.append({
            'Nome': nome, 'Ruolo': ruolo, 'Squadra': squadra, 'Media_Voto_Totale': round(ultima_media, 2),
            'Fattore_Campo': fattore_campo, 'Avversario': avversario, 'Difficolta_Ruolo': difficolta_ruolo,
            'Rigori': row.get('Rigori', 0), 'Punizioni': row.get('Punizioni', 0),
            'xG': float(row.get('xG', 0.0)), 'xA': float(row.get('xA', 0.0)), 'Fatica': float(row.get('Fatica', 0))
        })

    df_out = pd.DataFrame(prossima_giornata)
    if df_out.empty:
        return df_out, []

    # Calcolo Vettoriale delle Previsioni (molto più veloce di .apply)
    modelli = allena_modelli_specializzati()
    mask_def = df_out['Ruolo'].isin(['P', 'D'])
    
    df_out['Voto_Base_Puro'] = 0.0
    df_out['Bonus_Atteso'] = 0.0
    df_out['Err_Std'] = 0.0

    if mask_def.any():
        df_def = df_out[mask_def]
        df_out.loc[mask_def, 'Voto_Base_Puro'] = modelli['def_base'].predict(df_def)
        df_out.loc[mask_def, 'Bonus_Atteso'] = modelli['def_bonus'].predict(df_def)
        df_out.loc[mask_def, 'Err_Std'] = modelli['def_base'].scale + (df_def['xG'] * 1.5)

    if (~mask_def).any():
        df_att = df_out[~mask_def]
        df_out.loc[~mask_def, 'Voto_Base_Puro'] = modelli['att_base'].predict(df_att)
        df_out.loc[~mask_def, 'Bonus_Atteso'] = modelli['att_bonus'].predict(df_att)
        df_out.loc[~mask_def, 'Err_Std'] = np.sqrt(modelli['att_base'].mse_resid) + (df_att['xG'] * 2.0)

    df_out['xFP'] = (df_out['Voto_Base_Puro'] + df_out['Bonus_Atteso']).round(2)
    df_out['Voto_Base_Puro'] = df_out['Voto_Base_Puro'].round(2)
    df_out['Min_V'] = (df_out['xFP'] - df_out['Err_Std']).round(2)
    df_out['Max_V'] = (df_out['xFP'] + df_out['Err_Std']).round(2)
    df_out['VORP'] = (df_out['xFP'] - df_out['Ruolo'].map(BASELINE_RUOLO).fillna(5.5)).round(2)

    return df_out, []

# --- 4. MOTORE LOGICO E OTTIMIZZATORE VORP ---
def suggerisci_miglior_formazione(df_disponibili):
    df_ordinato = df_disponibili.sort_values(by=['Ruolo', 'xFP'], ascending=[True, False])
    candidati = []

    for modulo in MODULI_VALIDI:
        titolari_modulo, panchina_modulo, valore_modulo_vorp, modulo_valido, scelti_per_ruolo = [], [], 0, True, {}
        for ruolo in ['P', 'D', 'C', 'A']:
            disp = df_ordinato[df_ordinato['Ruolo'] == ruolo]
            n_richiesti = modulo[ruolo]
            if len(disp) < n_richiesti:
                modulo_valido = False
                break
            scelti = disp.head(n_richiesti)
            riserve = disp.iloc[n_richiesti:n_richiesti+2] # Primi 2 panchinari per ruolo
            titolari_modulo.append(scelti)
            panchina_modulo.append(riserve)
            scelti_per_ruolo[ruolo] = scelti
            valore_modulo_vorp += scelti['VORP'].sum()
            
        if modulo_valido:
            bonus_mod = 0
            if modulo['D'] >= 4:
                voto_p = scelti_per_ruolo['P']['Voto_Base_Puro'].values[0]
                top_3_d = scelti_per_ruolo['D'].sort_values(by='Voto_Base_Puro', ascending=False).head(3)['Voto_Base_Puro'].sum()
                media_mod = (voto_p + top_3_d) / 4.0
                bonus_mod = 6 if media_mod >= 7.0 else (3 if media_mod >= 6.5 else (1 if media_mod >= 6.0 else 0))
            
            punteggio_totale_vorp = valore_modulo_vorp + bonus_mod
            undici = pd.concat(titolari_modulo)
            panchina = pd.concat(panchina_modulo)
            punti_tot = undici['xFP'].sum() + bonus_mod

            candidati.append({
                'modulo': modulo['nome'],
                'punti_tot': punti_tot,
                'titolari': undici,
                'panchina': panchina,
                'bonus_mod': bonus_mod,
                'vorp_score': punteggio_totale_vorp
            })
            
    candidati.sort(key=lambda x: x['vorp_score'], reverse=True)
    primo = candidati[0] if len(candidati) > 0 else None
    secondo = candidati[1] if len(candidati) > 1 else None
    return primo, secondo

def renderizza_formazione(formazione_dict):
    undici = formazione_dict['titolari']
    panchina = formazione_dict['panchina']
    
    cP, cD, cC, cA = st.columns(4)
    ruoli_info = [
        (cP, 'P', 'Portiere', '🧤'),
        (cD, 'D', 'Difesa', '🛡️'),
        (cC, 'C', 'Centrocampo', '👟'),
        (cA, 'A', 'Attacco', '⚽')
    ]
    for col, codice_r, titolo_r, icona in ruoli_info:
        with col:
            st.subheader(titolo_r)
            for _, g in undici[undici['Ruolo'] == codice_r].iterrows():
                casa_trasf = "🏠" if g['Fattore_Campo'] == 1 else "✈️"
                st.info(f"{icona} **{g['Nome']}** ({g['xFP']})\n\n{casa_trasf} vs {g['Avversario']}")
                
    if not panchina.empty:
        with st.expander("🪑 Visualizza Panchina Consigliata (in ordine di priorità)"):
             pP, pD, pC, pA = st.columns(4)
             for col_p, codice_r, titolo_r, icona in zip([pP, pD, pC, pA], ['P', 'D', 'C', 'A'], ['Portieri', 'Difensori', 'Centrocampisti', 'Attaccanti'], ['🧤', '🛡️', '👟', '⚽']):
                 with col_p:
                     st.markdown(f"**{titolo_r}**")
                     ris_r = panchina[panchina['Ruolo'] == codice_r]
                     if ris_r.empty:
                         st.caption("Nessuna riserva")
                     for _, g in ris_r.iterrows():
                         st.write(f"{icona} {g['Nome']} ({g['xFP']})")

# --- 5. INTERFACCIA VISIVA ---
df_giornata, file_mancanti = prepara_dati_giornata(giornata_input)

if file_mancanti:
    st.error(f"⚠️ Impossibile avviare il motore: mancano i seguenti file nella cartella: **{', '.join(file_mancanti)}**")
elif df_giornata.empty:
    st.warning(f"⚠️ Nessuna partita trovata nel calendario per la **Giornata {giornata_input}** o nessuna squadra dei tuoi giocatori corrisponde.")
else:
    lista_tutti_giocatori = df_giornata['Nome'].tolist()
    
    # Filtro Indisponibili in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚑 Gestione Indisponibili")
    indisponibili = st.sidebar.multiselect("Escludi infortunati / squalificati:", lista_tutti_giocatori)
    
    df_attivi = df_giornata[~df_giornata['Nome'].isin(indisponibili)].copy()
    lista_giocatori = df_attivi['Nome'].tolist()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "👤 Analisi Singola", "⚔️ Ballottaggio", "🏆 Miglior Modulo", "🥈 2° Miglior Modulo", "📋 Rosa Completa"
    ])

    top_formazione, seconda_formazione = suggerisci_miglior_formazione(df_attivi)

    with tab1:
        st.header("Analisi Predittiva Avanzata")
        if not lista_giocatori:
            st.warning("Tutti i giocatori sono stati esclusi.")
        else:
            g_sel = st.selectbox("Seleziona Giocatore", lista_giocatori)
            dati_g = df_attivi[df_attivi['Nome'] == g_sel].iloc[0]
            
            campo_str = "In Casa 🏠" if dati_g['Fattore_Campo'] == 1 else "In Trasferta ✈️"
            st.info(f"🏟️ **Partita:** {dati_g['Squadra']} vs {dati_g['Avversario']} ({campo_str}) | **Difficoltà Match:** {dati_g['Difficolta_Ruolo']}/5 | **Min. Giocati (Fatica):** {int(dati_g['Fatica'])}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Fantapunti Attesi (xFP)", f"{dati_g['xFP']}")
            c2.metric("Voto Base Puro", f"{dati_g['Voto_Base_Puro']}")
            c3.metric("Peggior Caso Stimato", f"{dati_g['Min_V']}")
            c4.metric("Miglior Caso Stimato", f"{dati_g['Max_V']}")

    with tab2:
        st.header("Ballottaggio Titolare")
        if len(lista_giocatori) < 2:
            st.warning("Servono almeno 2 giocatori disponibili per un ballottaggio.")
        else:
            cA, cB = st.columns(2)
            with cA: g_A = st.selectbox("Giocatore A", lista_giocatori, index=0)
            with cB: g_B = st.selectbox("Giocatore B", lista_giocatori, index=1)
            
            d_A = df_attivi[df_attivi['Nome'] == g_A].iloc[0]
            d_B = df_attivi[df_attivi['Nome'] == g_B].iloc[0]
            
            st.divider()
            consigliato = g_A if d_A['xFP'] >= d_B['xFP'] else g_B
            scarto_bal = abs(d_A['xFP'] - d_B['xFP'])
            st.success(f"💡 Il Modello consiglia: **{consigliato}** (+{scarto_bal:.2f} xFP)")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[d_B['xFP'], d_A['xFP']], y=[g_B, g_A], mode='markers',
                marker=dict(color=['#ff7f0e', '#1f77b4'], size=16),
                error_x=dict(
                    type='data', symmetric=False,
                    array=[d_B['Max_V'] - d_B['xFP'], d_A['Max_V'] - d_A['xFP']],
                    arrayminus=[d_B['xFP'] - d_B['Min_V'], d_A['xFP'] - d_A['Min_V']],
                    color='gray', thickness=3, width=8
                )
            ))
            fig.update_layout(xaxis_title="Fantapunti Attesi (xFP) e Forbice di Rischio", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
    with tab3:
        st.header("1ª Scelta: La Formazione Ideale")
        if not top_formazione:
            st.warning("⚠️ Giocatori disponibili insufficienti per comporre un modulo valido.")
        else:
            bonus = top_formazione['bonus_mod']
            msg_b = f"(di cui +{bonus} da Modificatore Difesa)" if bonus > 0 else "(Nessun bonus modificatore)"
            st.success(f"🔥 Miglior Modulo: **{top_formazione['modulo']}** | Punti Attesi: **{top_formazione['punti_tot']:.2f}** {msg_b}")
            renderizza_formazione(top_formazione)

    with tab4:
        st.header("2ª Scelta: Il Piano B Tattico")
        if not seconda_formazione:
            st.warning("⚠️ Non ci sono abbastanza giocatori per comporre un secondo modulo alternativo.")
        else:
            diff_punti = top_formazione['punti_tot'] - seconda_formazione['punti_tot']
            bonus2 = seconda_formazione['bonus_mod']
            msg2 = f"(di cui +{bonus2} da Modificatore Difesa)" if bonus2 > 0 else "(Nessun bonus modificatore)"
            st.info(f"🥈 Modulo Alternativo: **{seconda_formazione['modulo']}** | Punti Attesi: **{seconda_formazione['punti_tot']:.2f}** {msg2} | Scarto dal 1°: **-{diff_punti:.2f} pt**")
            renderizza_formazione(seconda_formazione)

    with tab5:
        st.header("Riepilogo Completo Rosa di Giornata")
        df_view = df_attivi[['Nome', 'Ruolo', 'Squadra', 'Avversario', 'Fattore_Campo', 'Difficolta_Ruolo', 'Media_Voto_Totale', 'Voto_Base_Puro', 'xFP', 'VORP']].copy()
        df_view['Campo'] = df_view['Fattore_Campo'].map({1: 'Casa 🏠', 0: 'Trasf ✈️'})
        df_view = df_view.drop(columns=['Fattore_Campo']).sort_values(by='xFP', ascending=False)
        st.dataframe(df_view, use_container_width=True, hide_index=True)
