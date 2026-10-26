import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import plotly.graph_objects as go
import os
import unicodedata

st.set_page_config(page_title="Fantacalcio Pro", layout="wide")[cite: 7]
st.title("⚽ Fantacalcio: Motore Inferenziale Avanzato")[cite: 7]

# --- 1. MATRICE DIFENSIVA IBRIDA (Dinamica + Statica) ---
DIFFICOLTA_GRANULARE = {
    'Inter': {'P': 4, 'D': 5, 'C': 4, 'A': 5}, 'Juventus': {'P': 4, 'D': 4, 'C': 5, 'A': 5},
    'Napoli': {'P': 4, 'D': 4, 'C': 4, 'A': 4}, 'Atalanta': {'P': 4, 'D': 4, 'C': 4, 'A': 4},
    'Milan': {'P': 3, 'D': 4, 'C': 4, 'A': 4}, 'Roma': {'P': 3, 'D': 4, 'C': 4, 'A': 3},
    'Lazio': {'P': 3, 'D': 3, 'C': 4, 'A': 3}, 'Fiorentina':{'P': 3, 'D': 3, 'C': 3, 'A': 3},
    'Torino': {'P': 2, 'D': 3, 'C': 3, 'A': 3}, 'Bologna': {'P': 2, 'D': 3, 'C': 3, 'A': 3},
    'Udinese': {'P': 2, 'D': 2, 'C': 2, 'A': 2}, 'Genoa': {'P': 2, 'D': 2, 'C': 2, 'A': 2},
    'Monza': {'P': 2, 'D': 2, 'C': 2, 'A': 2}, 'Lecce': {'P': 2, 'D': 2, 'C': 2, 'A': 2},
    'Empoli': {'P': 2, 'D': 2, 'C': 2, 'A': 2}, 'Verona': {'P': 2, 'D': 2, 'C': 2, 'A': 2},
    'Cagliari': {'P': 1, 'D': 1, 'C': 1, 'A': 1}, 'Parma': {'P': 1, 'D': 1, 'C': 1, 'A': 1},
    'Como': {'P': 1, 'D': 1, 'C': 1, 'A': 1}, 'Venezia': {'P': 1, 'D': 1, 'C': 1, 'A': 1}
}[cite: 7]

try:
    df_classifica = pd.read_csv('classifica.csv', sep=';')[cite: 7]
    soglie_gf = df_classifica['GF'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()[cite: 7]
    soglie_gs = df_classifica['GS'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()[cite: 7]

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
            return 1[cite: 7]

    DIFFICOLTA_GRANULARE_DINAMICA = {}[cite: 7]
    for _, row in df_classifica.iterrows():
        sq = row['Squadra'].strip()[cite: 7]
        diff_pd = assegna_difficolta(row['GF'], soglie_gf, inverti=False)[cite: 7]
        diff_ca = assegna_difficolta(row['GS'], soglie_gs, inverti=True)[cite: 7]
        DIFFICOLTA_GRANULARE_DINAMICA[sq] = {'P': diff_pd, 'D': diff_pd, 'C': diff_ca, 'A': diff_ca}[cite: 7]
    DIFFICOLTA_GRANULARE = DIFFICOLTA_GRANULARE_DINAMICA[cite: 7]
except Exception:
    pass [cite: 7]

MODULI_VALIDI = [
    {'nome': '3-4-3', 'P': 1, 'D': 3, 'C': 4, 'A': 3}, {'nome': '3-5-2', 'P': 1, 'D': 3, 'C': 5, 'A': 2},
    {'nome': '4-3-3', 'P': 1, 'D': 4, 'C': 3, 'A': 3}, {'nome': '4-4-2', 'P': 1, 'D': 4, 'C': 4, 'A': 2},
    {'nome': '4-5-1', 'P': 1, 'D': 4, 'C': 5, 'A': 1}, {'nome': '5-3-2', 'P': 1, 'D': 5, 'C': 3, 'A': 2},
    {'nome': '5-4-1', 'P': 1, 'D': 5, 'C': 4, 'A': 1}
][cite: 7]
BASELINE_RUOLO = {'P': 5.0, 'D': 5.5, 'C': 6.0, 'A': 6.5}[cite: 7]

# --- 2. ZONA CARICAMENTO DATI (SIDEBAR) ---
st.sidebar.markdown("---")[cite: 7]
st.sidebar.subheader("🔄 Dati Giocatori (FBref)")[cite: 7]
file_fbref = st.sidebar.file_uploader("Trascina statistiche giocatori", type=['csv', 'txt', 'xls'])[cite: 7]

if file_fbref is not None:
    with st.spinner("Analisi inferenziale del file in corso..."):[cite: 7]
        try:
            for separatore in [',', ';']:
                try:
                    file_fbref.seek(0)[cite: 7]
                    df_utili = pd.read_csv(file_fbref, sep=separatore, on_bad_lines='skip', encoding='utf-8')[cite: 7]
                except UnicodeDecodeError:
                    file_fbref.seek(0)[cite: 7]
                    df_utili = pd.read_csv(file_fbref, sep=separatore, on_bad_lines='skip', encoding='latin-1')[cite: 7]
                colonne_trovate = df_utili.columns.tolist()[cite: 7]
                if 'Giocatore' not in colonne_trovate and 'Player' not in colonne_trovate:
                    try:
                        file_fbref.seek(0)[cite: 7]
                        df_utili = pd.read_csv(file_fbref, sep=separatore, skiprows=1, on_bad_lines='skip', encoding='utf-8')[cite: 7]
                    except UnicodeDecodeError:
                        file_fbref.seek(0)[cite: 7]
                        df_utili = pd.read_csv(file_fbref, sep=separatore, skiprows=1, on_bad_lines='skip', encoding='latin-1')[cite: 7]
                if 'Player' in df_utili.columns: df_utili = df_utili.rename(columns={'Player': 'Giocatore'})[cite: 7]
                if 'Giocatore' in df_utili.columns: break[cite: 7]
            
            if 'Giocatore' in df_utili.columns:
                xg_col = 'xG' if 'xG' in df_utili.columns else None[cite: 7]
                xag_col = 'xAG' if 'xAG' in df_utili.columns else 'xA' if 'xA' in df_utili.columns else None[cite: 7]
                
                if not xg_col and 'Gls' in df_utili.columns:
                    df_utili['xG_stimato'] = df_utili['Gls'].astype(str).str.replace(',', '.')[cite: 7]
                    xg_col = 'xG_stimato'[cite: 7]
                if not xag_col and 'Ast' in df_utili.columns:
                    df_utili['xA_stimato'] = df_utili['Ast'].astype(str).str.replace(',', '.')[cite: 7]
                    xag_col = 'xA_stimato'[cite: 7]

                if xg_col and xag_col:
                    df_utili = df_utili[['Giocatore', xg_col, xag_col]].copy()[cite: 7]
                    df_utili = df_utili[df_utili['Giocatore'] != 'Giocatore'][cite: 7]
                    df_utili['xG'] = pd.to_numeric(df_utili[xg_col], errors='coerce').fillna(0.0)[cite: 7]
                    df_utili['xA'] = pd.to_numeric(df_utili[xag_col], errors='coerce').fillna(0.0)[cite: 7]
                    
                    if os.path.exists('giocatori.csv'):
                        mio_file = pd.read_csv('giocatori.csv', sep=';').fillna(0)[cite: 7]
                        def pulisci_nome(nome):
                            if not isinstance(nome, str): return ""
                            return unicodedata.normalize('NFKD', nome).encode('ASCII', 'ignore').decode('utf-8').lower()[cite: 7]
                        df_utili['Giocatore_Clean'] = df_utili['Giocatore'].apply(pulisci_nome)[cite: 7]
                        
                        def trova_xg_xa(nome_giocatore):
                            match = df_utili[df_utili['Giocatore_Clean'].str.contains(pulisci_nome(nome_giocatore), case=False, na=False)][cite: 7]
                            if not match.empty: return match['xG'].values[0], match['xA'].values[0][cite: 7]
                            return 0.0, 0.0[cite: 7]
                        
                        risultati = mio_file['Nome'].apply(trova_xg_xa)[cite: 7]
                        mio_file['xG'], mio_file['xA'] = [res[0] for res in risultati], [res[1] for res in risultati][cite: 7]
                        mio_file.to_csv('giocatori.csv', sep=';', index=False)[cite: 7]
                        st.sidebar.success("✅ Dati offensivi aggiornati!")[cite: 7]
                    else: st.sidebar.error("Manca il file 'giocatori.csv'.")[cite: 7]
                else: st.sidebar.error("File privo di metriche xG/xA o Gol/Assist.")[cite: 7]
            else: st.sidebar.error("Colonna 'Giocatore' non trovata.")[cite: 7]
        except Exception as e: st.sidebar.error(f"Errore tecnico: {e}")[cite: 7]

st.sidebar.markdown("---")[cite: 7]
st.sidebar.subheader("📈 Dati Squadre (FBref)")[cite: 7]
file_classifica_upload = st.sidebar.file_uploader("Trascina statistiche Squadre", type=['csv', 'xls', 'txt'])[cite: 7]

if file_classifica_upload is not None:
    try:
        df_c = None[cite: 7]
        if file_classifica_upload.name.lower().endswith(('.xls', '.html')):
            try:
                file_classifica_upload.seek(0)[cite: 7]
                dfs = pd.read_html(file_classifica_upload)[cite: 7]
                if dfs:
                    df_c = dfs[0][cite: 7]
                    if isinstance(df_c.columns, pd.MultiIndex):
                        df_c.columns = df_c.columns.get_level_values(-1)[cite: 7]
            except Exception:
                pass[cite: 7]
                
        if df_c is None:
            for sep in [',', ';']:
                try:
                    file_classifica_upload.seek(0)[cite: 7]
                    temp_df = pd.read_csv(file_classifica_upload, sep=sep, on_bad_lines='skip', encoding='utf-8')[cite: 7]
                except UnicodeDecodeError:
                    file_classifica_upload.seek(0)[cite: 7]
                    temp_df = pd.read_csv(file_classifica_upload, sep=sep, on_bad_lines='skip', encoding='latin-1')[cite: 7]
                
                if 'Squad' not in temp_df.columns and 'Squadra' not in temp_df.columns:
                    try:
                        file_classifica_upload.seek(0)[cite: 7]
                        temp_df = pd.read_csv(file_classifica_upload, sep=sep, skiprows=1, on_bad_lines='skip', encoding='utf-8')[cite: 7]
                    except UnicodeDecodeError:
                        file_classifica_upload.seek(0)[cite: 7]
                        temp_df = pd.read_csv(file_classifica_upload, sep=sep, skiprows=1, on_bad_lines='skip', encoding='latin-1')[cite: 7]
                
                if 'Squad' in temp_df.columns or 'Squadra' in temp_df.columns:
                    df_c = temp_df[cite: 7]
                    break[cite: 7]
                
        if df_c is not None:
            col_squad = 'Squad' if 'Squad' in df_c.columns else 'Squadra'[cite: 7]
            col_gf = 'Gls' if 'Gls' in df_c.columns else 'GF' if 'GF' in df_c.columns else None[cite: 7]
            col_gs = 'GA' if 'GA' in df_c.columns else 'GS' if 'GS' in df_c.columns else None[cite: 7]
            
            if col_gf and col_gs:
                df_clean = pd.DataFrame()[cite: 7]
                df_clean['Squadra'] = df_c[col_squad].astype(str).str.replace(r'^[a-z]{2}\s+', '', regex=True).str.strip()[cite: 7]
                df_clean['GF'] = pd.to_numeric(df_c[col_gf], errors='coerce').fillna(0)[cite: 7]
                df_clean['GS'] = pd.to_numeric(df_c[col_gs], errors='coerce').fillna(0)[cite: 7]
                
                df_clean = df_clean[df_clean['Squadra'] != 'nan'][cite: 7]
                df_clean.to_csv('classifica.csv', sep=';', index=False)[cite: 7]
                st.sidebar.success("✅ Difficoltà Dinamiche ricalcolate da FBref! Ricarica la pagina.")[cite: 7]
            else:
                st.sidebar.error("❌ Il file non contiene le colonne Gls (Gol Fatti) e GA (Gol Subiti).")[cite: 7]
        else:
            st.sidebar.error("❌ Colonna Squad/Squadra non trovata.")[cite: 7]
    except Exception as e: 
        st.sidebar.error(f"Errore tecnico: {e}")[cite: 7]

# --- 3. PREPARAZIONE DATI E MODELLI INFERENZIALI ---
@st.cache_data
def prepara_dati_giornata(giornata_da_giocare):
    file_richiesti = ['giocatori.csv', 'calendario.csv', 'storico_voti_base.csv', 'storico_bonus.csv'][cite: 7]
    if not all(os.path.exists(f) for f in file_richiesti): return pd.DataFrame()[cite: 7]
            
    giocatori = pd.read_csv('giocatori.csv', sep=';').fillna(0)[cite: 7]
    calendario = pd.read_csv('calendario.csv', sep=';')[cite: 7]
    v_base = pd.read_csv('storico_voti_base.csv', sep=';').melt(id_vars=['Nome'], var_name='Giornata', value_name='Voto_Base')[cite: 7]
    v_bonus = pd.read_csv('storico_bonus.csv', sep=';').melt(id_vars=['Nome'], var_name='Giornata', value_name='Bonus')[cite: 7]
    
    v_base['Giornata'] = v_base['Giornata'].astype(str).str.replace('G', '').astype(int)[cite: 7]
    v_bonus['Giornata'] = v_bonus['Giornata'].astype(str).str.replace('G', '').astype(int)[cite: 7]
    
    storico = pd.merge(v_base, v_bonus, on=['Nome', 'Giornata'], how='outer')[cite: 7]
    storico['Voto_Base'] = pd.to_numeric(storico['Voto_Base'], errors='coerce')[cite: 7]
    storico['Bonus'] = pd.to_numeric(storico['Bonus'], errors='coerce').fillna(0.0)[cite: 7]
    
    df_completo = pd.merge(storico.dropna(subset=['Voto_Base']), giocatori, on='Nome', how='left').sort_values(by=['Nome', 'Giornata'])[cite: 7]
    
    df_completo['Media_Voto_Totale'] = df_completo.groupby('Nome')['Voto_Base'].transform(
        lambda x: x.shift(1).ewm(span=5, min_periods=1).mean().fillna(5.0)
    )[cite: 7]
    
    partite_oggi = calendario[calendario['Giornata'] == giornata_da_giocare][cite: 7]
    prossima_giornata = [][cite: 7]

    for _, row in giocatori.iterrows():
        nome, squadra, ruolo = row['Nome'], row['Squadra'].strip(), row['Ruolo'][cite: 7]
        storico_g = df_completo[df_completo['Nome'] == nome][cite: 7]
        ultima_media = storico_g['Media_Voto_Totale'].iloc[-1] if not storico_g.empty else 5.0[cite: 7]
        
        match_casa = partite_oggi[partite_oggi['Squadra_Casa'].str.strip() == squadra][cite: 7]
        match_trasf = partite_oggi[partite_oggi['Squadra_Trasferta'].str.strip() == squadra][cite: 7]
        
        if not match_casa.empty: fattore_campo, avversario = 1, match_casa['Squadra_Trasferta'].values[0].strip()[cite: 7]
        elif not match_trasf.empty: fattore_campo, avversario = 0, match_trasf['Squadra_Casa'].values[0].strip()[cite: 7]
        else: continue[cite: 7]
            
        difficolta_ruolo = DIFFICOLTA_GRANULARE.get(avversario, {'P': 3, 'D': 3, 'C': 3, 'A': 3})[ruolo][cite: 7]
        
        prossima_giornata.append({
            'Nome': nome, 'Ruolo': ruolo, 'Squadra': squadra, 'Media_Voto_Totale': round(ultima_media, 2),
            'Fattore_Campo': fattore_campo, 'Avversario': avversario, 'Difficolta_Ruolo': difficolta_ruolo,
            'Rigori': row.get('Rigori', 0), 'Punizioni': row.get('Punizioni', 0),
            'xG': row.get('xG', 0.0), 'xA': row.get('xA', 0.0), 'Fatica': row.get('Fatica', 0)
        })[cite: 7]

    return pd.DataFrame(prossima_giornata)[cite: 7]

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
    })[cite: 7]
    
    form_base = 'Voto_Base ~ Media_Voto_Totale + Fatica + Difficolta_Ruolo * Fattore_Campo'[cite: 7]
    form_bonus = 'Bonus ~ I(xG * 3 + xA * 1) + Rigori + Punizioni + Difficolta_Ruolo * Fattore_Campo'[cite: 7]
    
    train_def = dati_train[dati_train['Ruolo'].isin(['P', 'D'])].copy()[cite: 7]
    mod_base_def = smf.rlm(formula=form_base, data=train_def).fit()[cite: 7]
    mod_bonus_def = smf.rlm(formula=form_bonus, data=train_def).fit()[cite: 7]
    
    train_att = dati_train[dati_train['Ruolo'].isin(['C', 'A'])].copy()[cite: 7]
    mod_base_att = smf.ols(formula=form_base, data=train_att).fit()[cite: 7]
    mod_bonus_att = smf.ols(formula=form_bonus, data=train_att).fit()[cite: 7]
    
    return {'def_base': mod_base_def, 'def_bonus': mod_bonus_def, 'att_base': mod_base_att, 'att_bonus': mod_bonus_att}[cite: 7]

# --- 4. MOTORE LOGICO E OTTIMIZZATORE VORP ---
giornata_input = st.sidebar.number_input("Imposta Giornata", min_value=1, max_value=38, value=3)[cite: 7]
df_giornata = prepara_dati_giornata(giornata_input)[cite: 7]

if not df_giornata.empty:
    modelli = allena_modelli_specializzati()[cite: 7]

def calcola_previsioni(dati_giocatore):
    ruolo = dati_giocatore['Ruolo'].values[0][cite: 7]
    
    if ruolo in ['P', 'D']:
        v_base = modelli['def_base'].predict(dati_giocatore).values[0][cite: 7]
        bonus = modelli['def_bonus'].predict(dati_giocatore).values[0][cite: 7]
        err_std = modelli['def_base'].scale + (dati_giocatore['xG'].values[0] * 1.5)[cite: 7]
    else:
        v_base = modelli['att_base'].predict(dati_giocatore).values[0][cite: 7]
        bonus = modelli['att_bonus'].predict(dati_giocatore).values[0][cite: 7]
        err_std = np.sqrt(modelli['att_base'].mse_resid) + (dati_giocatore['xG'].values[0] * 2.0)[cite: 7]
        
    xfp_tot = v_base + bonus[cite: 7]
    min_v, max_v = xfp_tot - err_std, xfp_tot + err_std[cite: 7]
    return round(xfp_tot, 2), round(v_base, 2), round(min_v, 2), round(max_v, 2)[cite: 7]

def suggerisci_miglior_formazione(df_giornata):
    previsioni = df_giornata.apply(lambda x: calcola_previsioni(pd.DataFrame([x])), axis=1)[cite: 7]
    df_giornata['xFP'] = [p[0] for p in previsioni][cite: 7]
    df_giornata['Voto_Base_Puro'] = [p[1] for p in previsioni][cite: 7]
    
    df_giornata['VORP'] = df_giornata.apply(lambda r: r['xFP'] - BASELINE_RUOLO.get(r['Ruolo'], 5.5), axis=1)[cite: 7]
    df_ordinato = df_giornata.sort_values(by=['Ruolo', 'xFP'], ascending=[True, False])[cite: 7]
    
    candidati = []

    for modulo in MODULI_VALIDI:[cite: 7]
        titolari_modulo, valore_modulo_vorp, modulo_valido, scelti_per_ruolo = [], 0, True, {}[cite: 7]
        for ruolo in ['P', 'D', 'C', 'A']:[cite: 7]
            disp = df_ordinato[df_ordinato['Ruolo'] == ruolo][cite: 7]
            if len(disp) < modulo[ruolo]:[cite: 7]
                modulo_valido = False[cite: 7]
                break[cite: 7]
            scelti = disp.head(modulo[ruolo])[cite: 7]
            titolari_modulo.append(scelti)[cite: 7]
            scelti_per_ruolo[ruolo] = scelti[cite: 7]
            valore_modulo_vorp += scelti['VORP'].sum()[cite: 7]
            
        if modulo_valido:
            bonus_mod = 0[cite: 7]
            if modulo['D'] >= 4:[cite: 7]
                media_mod = (scelti_per_ruolo['P']['Voto_Base_Puro'].values[0] + sum(scelti_per_ruolo['D'].sort_values(by='Voto_Base_Puro', ascending=False).head(3)['Voto_Base_Puro'].tolist())) / 4[cite: 7]
                bonus_mod = 6 if media_mod >= 7.0 else (3 if media_mod >= 6.5 else (1 if media_mod >= 6.0 else 0))[cite: 7]
            
            punteggio_totale_vorp = valore_modulo_vorp + bonus_mod
            undici = pd.concat(titolari_modulo)
            punti_tot = undici['xFP'].sum() + bonus_mod

            candidati.append({
                'modulo': modulo['nome'],
                'punti_tot': punti_tot,
                'titolari': undici,
                'bonus_mod': bonus_mod,
                'vorp_score': punteggio_totale_vorp
            })
            
    candidati.sort(key=lambda x: x['vorp_score'], reverse=True)
    
    primo = candidati[0] if len(candidati) > 0 else None
    secondo = candidati[1] if len(candidati) > 1 else None
    
    return primo, secondo

# --- 5. INTERFACCIA VISIVA ---
if not df_giornata.empty:
    lista_giocatori = df_giornata['Nome'].tolist()[cite: 7]
    tab1, tab2, tab3, tab4 = st.tabs(["👤 Analisi Singola", "⚔️ Ballottaggio", "🏆 Miglior Modulo", "🥈 2° Miglior Modulo"])

    top_formazione, seconda_formazione = suggerisci_miglior_formazione(df_giornata)

    with tab1:
        st.header("Analisi Predittiva Avanzata")[cite: 7]
        g_sel = st.selectbox("Giocatore", lista_giocatori)[cite: 7]
        dati_g = df_giornata[df_giornata['Nome'] == g_sel][cite: 7]
        xfp, vbase, min_v, max_v = calcola_previsioni(dati_g)[cite: 7]
        st.info(f"🏟️ **Contro:** {dati_g['Avversario'].values[0]} | **Difficoltà:** {dati_g['Difficolta_Ruolo'].values[0]}/5 | **Min. Giocati:** {dati_g['Fatica'].values[0]}")[cite: 7]
        c1, c2, c3, c4 = st.columns(4)[cite: 7]
        c1.metric("Atteso (xFP)", f"{xfp}")[cite: 7]
        c2.metric("Voto Puro", f"{vbase}")[cite: 7]
        c3.metric("Peggior Caso", f"{min_v}")[cite: 7]
        c4.metric("Miglior Caso", f"{max_v}")[cite: 7]

    with tab2:
        st.header("Ballottaggio Titolare")[cite: 7]
        cA, cB = st.columns(2)[cite: 7]
        with cA: g_A = st.selectbox("Giocatore A", lista_giocatori, index=0)[cite: 7]
        with cB: g_B = st.selectbox("Giocatore B", lista_giocatori, index=min(1, len(lista_giocatori)-1))[cite: 7]
        xfp_A, v_A, m_A, M_A = calcola_previsioni(df_giornata[df_giornata['Nome'] == g_A])[cite: 7]
        xfp_B, v_B, m_B, M_B = calcola_previsioni(df_giornata[df_giornata['Nome'] == g_B])[cite: 7]
        st.divider()[cite: 7]
        st.success(f"💡 Il Modello consiglia: **{g_A if xfp_A > xfp_B else g_B}**")[cite: 7]
        fig = go.Figure()[cite: 7]
        fig.add_trace(go.Scatter(x=[xfp_B, xfp_A], y=[g_B, g_A], mode='markers', marker=dict(color=['orange', 'blue'], size=15), error_x=dict(type='data', symmetric=False, array=[M_B-xfp_B, M_A-xfp_A], arrayminus=[xfp_B-m_B, xfp_A-m_A], color='gray', thickness=3, width=8)))[cite: 7]
        fig.update_layout(xaxis_title="Fantapunti Attesi (xFP)", height=300)[cite: 7]
        st.plotly_chart(fig, use_container_width=True)[cite: 7]
        
    with tab3:
        st.header("1ª Scelta: La Formazione Ideale")
        if not top_formazione:
            st.warning("⚠️ Mancano giocatori per comporre un modulo.")
        else:
            bonus = top_formazione['bonus_mod']
            msg_b = f"(di cui +{bonus} da Modificatore)" if bonus > 0 else "(No modificatore)"
            st.success(f"🔥 Miglior Modulo: **{top_formazione['modulo']}** | Punti Attesi: **{top_formazione['punti_tot']:.2f}** {msg_b}")
            
            top_11 = top_formazione['titolari']
            cP, cD, cC, cA = st.columns(4)[cite: 7]
            with cP: 
                st.subheader("Portiere")[cite: 7]
                for _, g in top_11[top_11['Ruolo']=='P'].iterrows(): st.info(f"🧤 **{g['Nome']}** ({g['xFP']})")[cite: 7]
            with cD:
                st.subheader("Difesa")[cite: 7]
                for _, g in top_11[top_11['Ruolo']=='D'].iterrows(): st.info(f"🛡️ **{g['Nome']}** ({g['xFP']})")[cite: 7]
            with cC:
                st.subheader("Centrocampo")[cite: 7]
                for _, g in top_11[top_11['Ruolo']=='C'].iterrows(): st.info(f"👟 **{g['Nome']}** ({g['xFP']})")[cite: 7]
            with cA:
                st.subheader("Attacco")[cite: 7]
                for _, g in top_11[top_11['Ruolo']=='A'].iterrows(): st.info(f"⚽ **{g['Nome']}** ({g['xFP']})")[cite: 7]

    with tab4:
        st.header("2ª Scelta: Il Piano B Tattico")
        if not seconda_formazione:
            st.warning("⚠️ Non ci sono abbastanza giocatori per comporre un secondo modulo alternativo.")
        else:
            diff_punti = top_formazione['punti_tot'] - seconda_formazione['punti_tot']
            bonus2 = seconda_formazione['bonus_mod']
            msg2 = f"(di cui +{bonus2} da Modificatore)" if bonus2 > 0 else "(No modificatore)"
            
            st.info(f"🥈 Modulo Alternativo: **{seconda_formazione['modulo']}** | Punti Attesi: **{seconda_formazione['punti_tot']:.2f}** {msg2} | Scarto: -{diff_punti:.2f} pt")
            
            undici_2 = seconda_formazione['titolari']
            cP2, cD2, cC2, cA2 = st.columns(4)
            with cP2:
                st.subheader("Portiere")
                for _, g in undici_2[undici_2['Ruolo']=='P'].iterrows(): st.info(f"🧤 **{g['Nome']}** ({g['xFP']})")
            with cD2:
                st.subheader("Difesa")
                for _, g in undici_2[undici_2['Ruolo']=='D'].iterrows(): st.info(f"🛡️ **{g['Nome']}** ({g['xFP']})")
            with cC2:
                st.subheader("Centrocampo")
                for _, g in undici_2[undici_2['Ruolo']=='C'].iterrows(): st.info(f"👟 **{g['Nome']}** ({g['xFP']})")
            with cA2:
                st.subheader("Attacco")
                for _, g in undici_2[undici_2['Ruolo']=='A'].iterrows(): st.info(f"⚽ **{g['Nome']}** ({g['xFP']})")
