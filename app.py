import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import os
import io

st.set_page_config(page_title="Fantacalcio AI Pro", layout="wide")
st.title("⚽ Fantacalcio AI: Modello Avanzato Predittivo")

# --- 1. MATRICE DIFENSIVA GRANULARE E MODULI ---
DIFFICOLTA_GRANULARE = {
    'Inter':     {'P': 4, 'D': 5, 'C': 4, 'A': 5},
    'Juventus':  {'P': 4, 'D': 4, 'C': 5, 'A': 5},
    'Napoli':    {'P': 4, 'D': 4, 'C': 4, 'A': 4},
    'Atalanta':  {'P': 4, 'D': 4, 'C': 4, 'A': 4},
    'Milan':     {'P': 3, 'D': 4, 'C': 4, 'A': 4},
    'Roma':      {'P': 3, 'D': 4, 'C': 4, 'A': 3},
    'Lazio':     {'P': 3, 'D': 3, 'C': 4, 'A': 3},
    'Fiorentina':{'P': 3, 'D': 3, 'C': 3, 'A': 3},
    'Torino':    {'P': 2, 'D': 3, 'C': 3, 'A': 3},
    'Bologna':   {'P': 2, 'D': 3, 'C': 3, 'A': 3},
    'Udinese':   {'P': 2, 'D': 2, 'C': 2, 'A': 2},
    'Genoa':     {'P': 2, 'D': 2, 'C': 2, 'A': 2},
    'Monza':     {'P': 2, 'D': 2, 'C': 2, 'A': 2},
    'Lecce':     {'P': 2, 'D': 2, 'C': 2, 'A': 2},
    'Empoli':    {'P': 2, 'D': 2, 'C': 2, 'A': 2},
    'Verona':    {'P': 2, 'D': 2, 'C': 2, 'A': 2},
    'Cagliari':  {'P': 1, 'D': 1, 'C': 1, 'A': 1},
    'Parma':     {'P': 1, 'D': 1, 'C': 1, 'A': 1},
    'Como':      {'P': 1, 'D': 1, 'C': 1, 'A': 1},
    'Venezia':   {'P': 1, 'D': 1, 'C': 1, 'A': 1}
}

MODULI_VALIDI = [
    {'nome': '3-4-3', 'P': 1, 'D': 3, 'C': 4, 'A': 3},
    {'nome': '3-5-2', 'P': 1, 'D': 3, 'C': 5, 'A': 2},
    {'nome': '4-3-3', 'P': 1, 'D': 4, 'C': 3, 'A': 3},
    {'nome': '4-4-2', 'P': 1, 'D': 4, 'C': 4, 'A': 2},
    {'nome': '4-5-1', 'P': 1, 'D': 4, 'C': 5, 'A': 1},
    {'nome': '5-3-2', 'P': 1, 'D': 5, 'C': 3, 'A': 2},
    {'nome': '5-4-1', 'P': 1, 'D': 5, 'C': 4, 'A': 1}
]

# --- 2. ZONA CARICAMENTO DATI (SIDEBAR) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Aggiornamento Dati Statistiche")
st.sidebar.markdown("Carica qui qualsiasi file esportato da FBref (anche se mancano xG/xA, l'IA farà una stima usando Gol e Assist reali).")

file_fbref = st.sidebar.file_uploader("Trascina qui il file", type=['csv', 'txt', 'xls'])

if file_fbref is not None:
    with st.spinner("Analisi intelligente del file in corso..."):
        try:
            # Prova prima con la virgola, poi con il punto e virgola
            for separatore in [',', ';']:
                try:
                    file_fbref.seek(0)
                    df_utili = pd.read_csv(file_fbref, sep=separatore, on_bad_lines='skip', encoding='utf-8')
                except UnicodeDecodeError:
                    file_fbref.seek(0)
                    df_utili = pd.read_csv(file_fbref, sep=separatore, on_bad_lines='skip', encoding='latin-1')
                
                # Cerca l'intestazione corretta, saltando righe se necessario
                colonne_trovate = df_utili.columns.tolist()
                if 'Giocatore' not in colonne_trovate and 'Player' not in colonne_trovate:
                    try:
                        file_fbref.seek(0)
                        df_utili = pd.read_csv(file_fbref, sep=separatore, skiprows=1, on_bad_lines='skip', encoding='utf-8')
                    except UnicodeDecodeError:
                        file_fbref.seek(0)
                        df_utili = pd.read_csv(file_fbref, sep=separatore, skiprows=1, on_bad_lines='skip', encoding='latin-1')
                
                # Rinomina per standardizzare
                if 'Player' in df_utili.columns:
                    df_utili = df_utili.rename(columns={'Player': 'Giocatore'})
                if 'Giocatore' in df_utili.columns:
                    break
            
            # --- MOTORE FLESSIBILE ---
            if 'Giocatore' in df_utili.columns:
                # 1. Cerca i dati predittivi perfetti (xG, xAG)
                xg_col = 'xG' if 'xG' in df_utili.columns else None
                xag_col = 'xAG' if 'xAG' in df_utili.columns else 'xA' if 'xA' in df_utili.columns else None
                usati_dati_stima = False

                # 2. Se non li trova, attiva il piano di riserva (Usa Gol e Assist reali)
                if not xg_col and 'Gls' in df_utili.columns:
                    df_utili['xG_stimato'] = df_utili['Gls'].astype(str).str.replace(',', '.')
                    xg_col = 'xG_stimato'
                    usati_dati_stima = True
                
                if not xag_col and 'Ast' in df_utili.columns:
                    df_utili['xA_stimato'] = df_utili['Ast'].astype(str).str.replace(',', '.')
                    xag_col = 'xA_stimato'
                    usati_dati_stima = True

                # 3. Aggiorna il database
                if xg_col and xag_col:
                    df_utili = df_utili[['Giocatore', xg_col, xag_col]].copy()
                    df_utili = df_utili[df_utili['Giocatore'] != 'Giocatore'] # Pulisce eventuali sporcizie FBref
                    
                    df_utili['xG'] = pd.to_numeric(df_utili[xg_col], errors='coerce').fillna(0.0)
                    df_utili['xA'] = pd.to_numeric(df_utili[xag_col], errors='coerce').fillna(0.0)
                    
                    if os.path.exists('giocatori.csv'):
                        mio_file = pd.read_csv('giocatori.csv', sep=';')
                        
                        def trova_xg_xa(nome_giocatore):
                            match = df_utili[df_utili['Giocatore'].str.contains(nome_giocatore, case=False, na=False)]
                            if not match.empty:
                                return match['xG'].values[0], match['xA'].values[0]
                            return 0.0, 0.0
                        
                        risultati = mio_file['Nome'].apply(trova_xg_xa)
                        mio_file['xG'] = [res[0] for res in risultati]
                        mio_file['xA'] = [res[1] for res in risultati]
                        mio_file.to_csv('giocatori.csv', sep=';', index=False)
                        
                        if usati_dati_stima:
                            st.sidebar.warning("⚠️ Dati aggiornati usando Gls/Ast come stima (xG/xAG non trovati).")
                        else:
                            st.sidebar.success("✅ Dati aggiornati perfettamente con metriche xG/xAG!")
                    else:
                        st.sidebar.error("Manca il file 'giocatori.csv' nella cartella.")
                else:
                    st.sidebar.error("Errore: Il file non contiene né xG/xAG né i Gol/Assist base.")
            else:
                st.sidebar.error("Non riesco a trovare la colonna 'Giocatore' nel file.")
        except Exception as e:
            st.sidebar.error(f"Errore tecnico: {e}")
st.sidebar.markdown("---")

# --- 3. PREPARAZIONE DATI E MODELLI ---
@st.cache_data
def prepara_dati_giornata(giornata_da_giocare):
    file_richiesti = ['giocatori.csv', 'calendario.csv', 'storico_voti_base.csv', 'storico_bonus.csv']
    for file in file_richiesti:
        if not os.path.exists(file):
            st.error(f"ERRORE: Manca il file '{file}' nella cartella!")
            return pd.DataFrame()
            
    giocatori = pd.read_csv('giocatori.csv', sep=';')
    calendario = pd.read_csv('calendario.csv', sep=';')
    voti_base_oriz = pd.read_csv('storico_voti_base.csv', sep=';')
    voti_bonus_oriz = pd.read_csv('storico_bonus.csv', sep=';')
    
    v_base = voti_base_oriz.melt(id_vars=['Nome'], var_name='Giornata', value_name='Voto_Base')
    v_bonus = voti_bonus_oriz.melt(id_vars=['Nome'], var_name='Giornata', value_name='Bonus')
    
    v_base['Giornata'] = v_base['Giornata'].astype(str).str.replace('G', '').astype(int)
    v_bonus['Giornata'] = v_bonus['Giornata'].astype(str).str.replace('G', '').astype(int)
    
    storico = pd.merge(v_base, v_bonus, on=['Nome', 'Giornata'], how='outer')
    storico['Voto_Base'] = pd.to_numeric(storico['Voto_Base'], errors='coerce')
    storico['Bonus'] = pd.to_numeric(storico['Bonus'], errors='coerce').fillna(0.0)
    
    storico = storico.dropna(subset=['Voto_Base'])
    df_completo = pd.merge(storico, giocatori, on='Nome', how='left').sort_values(by=['Nome', 'Giornata'])
    
    # Calcola la media voto pura su TUTTO lo storico stagionale
    df_completo['Media_Voto_Totale'] = df_completo.groupby('Nome')['Voto_Base'].transform(
        lambda x: x.shift(1).expanding().mean().fillna(5.0)
    )
    
    partite_oggi = calendario[calendario['Giornata'] == giornata_da_giocare]
    prossima_giornata = []

    for _, row in giocatori.iterrows():
        nome, squadra, ruolo = row['Nome'], row['Squadra'], row['Ruolo']
        rigori, punizioni = row.get('Rigori', 0), row.get('Punizioni', 0)
        xg, xa, fatica = row.get('xG', 0.0), row.get('xA', 0.0), row.get('Fatica', 0)
        
        storico_g = df_completo[df_completo['Nome'] == nome]
        ultima_media = storico_g['Media_Voto_Totale'].iloc[-1] if not storico_g.empty else 5.0
        
        match_casa = partite_oggi[partite_oggi['Squadra_Casa'] == squadra]
        match_trasf = partite_oggi[partite_oggi['Squadra_Trasferta'] == squadra]
        
        if not match_casa.empty:
            fattore_campo, avversario = 1, match_casa['Squadra_Trasferta'].values[0]
        elif not match_trasf.empty:
            fattore_campo, avversario = 0, match_trasf['Squadra_Casa'].values[0]
        else:
            continue
            
        difficolta_ruolo = DIFFICOLTA_GRANULARE.get(avversario, {'P': 3, 'D': 3, 'C': 3, 'A': 3})[ruolo]
        
        prossima_giornata.append({
            'Nome': nome, 'Ruolo': ruolo, 'Squadra': squadra,
            'Media_Voto_Totale': round(ultima_media, 2),
            'Fattore_Campo': fattore_campo, 'Avversario': avversario,
            'Difficolta_Ruolo': difficolta_ruolo,
            'Rigori': rigori, 'Punizioni': punizioni,
            'xG': xg, 'xA': xa, 'Fatica': fatica
        })

    return pd.DataFrame(prossima_giornata)

@st.cache_resource
def allena_modelli():
    dati_train = pd.DataFrame({
        'Voto_Base': [6.0, 5.0, 6.5, 6.0, 6.5, 5.5, 6.5, 7.0, 5.0, 6.0, 5.5, 6.5, 4.5, 6.0, 5.5, 6.5, 7.0, 4.0],
        'Media_Voto_Totale': [6.2, 5.5, 6.5, 6.0, 6.5, 5.8, 6.2, 6.8, 5.5, 6.2, 5.8, 6.5, 5.0, 6.0, 5.5, 6.2, 7.0, 4.5],
        'Bonus': [0.0, 0.0, 3.0, 0.5, 0.0, 0.0, 1.0, 3.0, -0.5, 0.0, -0.5, 1.0, -1.0, 0.0, -1.0, 0.0, 3.5, -2.0],
        'Fatica': [90, 90, 45, 0, 90, 75, 90, 90, 90, 90, 0, 45, 90, 90, 90, 90, 90, 90],
        'xG': [0.1, 0.0, 0.8, 0.3, 0.0, 0.1, 0.2, 0.5, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
        'xA': [0.0, 0.0, 0.1, 0.2, 0.1, 0.0, 0.4, 0.3, 0.0, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
        'Rigori': [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'Punizioni': [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'Difficolta_Ruolo': [4, 5, 2, 3, 5, 3, 2, 1, 5, 3, 4, 2, 5, 3, 4, 2, 1, 5],
        'Fattore_Campo': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        'Ruolo': ['A', 'A', 'A', 'A', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'P', 'P', 'P', 'P', 'A', 'P']
    })
    
    form_base = 'Voto_Base ~ Media_Voto_Totale + Fatica + Difficolta_Ruolo + Fattore_Campo'
    mod_base = smf.ols(formula=form_base, data=dati_train).fit()
    
    form_bonus = 'Bonus ~ xG + xA + Rigori + Punizioni + Difficolta_Ruolo + Fattore_Campo'
    mod_bonus = smf.ols(formula=form_bonus, data=dati_train).fit()
    
    return mod_base, mod_bonus

# --- 4. MOTORE LOGICO ---
giornata_input = st.sidebar.number_input("Imposta Giornata", min_value=1, max_value=38, value=3)
df_giornata = prepara_dati_giornata(giornata_input)
if not df_giornata.empty:
    modello_base, modello_bonus = allena_modelli()

def calcola_previsioni(dati_giocatore):
    prev_base = modello_base.get_prediction(dati_giocatore).summary_frame(alpha=0.05)
    prev_bonus = modello_bonus.get_prediction(dati_giocatore).summary_frame(alpha=0.05)
    
    v_base_atteso = prev_base['mean'].values[0]
    bonus_atteso = prev_bonus['mean'].values[0]
    xfp_tot = v_base_atteso + bonus_atteso
    
    min_v = prev_base['obs_ci_lower'].values[0] + prev_bonus['obs_ci_lower'].values[0]
    max_v = prev_base['obs_ci_upper'].values[0] + prev_bonus['obs_ci_upper'].values[0]
    
    return round(xfp_tot, 2), round(v_base_atteso, 2), round(min_v, 2), round(max_v, 2)

def suggerisci_miglior_formazione(df_giornata):
    previsioni = df_giornata.apply(lambda x: calcola_previsioni(pd.DataFrame([x])), axis=1)
    df_giornata['xFP'] = [p[0] for p in previsioni]
    df_giornata['Voto_Base_Puro'] = [p[1] for p in previsioni]
    
    df_ordinato = df_giornata.sort_values(by=['Ruolo', 'xFP'], ascending=[True, False])
    
    miglior_modulo, miglior_punteggio_totale = None, 0
    migliori_undici = pd.DataFrame()
    dettagli_modificatore = 0 

    for modulo in MODULI_VALIDI:
        titolari_modulo = []
        punteggio_modulo, modulo_valido = 0, True
        scelti_per_ruolo = {}
        
        for ruolo in ['P', 'D', 'C', 'A']:
            giocatori_disponibili = df_ordinato[df_ordinato['Ruolo'] == ruolo]
            if len(giocatori_disponibili) < modulo[ruolo]:
                modulo_valido = False
                break
            scelti = giocatori_disponibili.head(modulo[ruolo])
            titolari_modulo.append(scelti)
            scelti_per_ruolo[ruolo] = scelti
            punteggio_modulo += scelti['xFP'].sum()
            
        if modulo_valido:
            bonus_mod = 0
            if modulo['D'] >= 4:
                voto_p = scelti_per_ruolo['P']['Voto_Base_Puro'].values[0]
                difensori_puri = scelti_per_ruolo['D'].sort_values(by='Voto_Base_Puro', ascending=False)
                voti_d = difensori_puri.head(3)['Voto_Base_Puro'].tolist()
                
                media_mod = (voto_p + sum(voti_d)) / 4
                if media_mod >= 7.0: bonus_mod = 6
                elif media_mod >= 6.5: bonus_mod = 3
                elif media_mod >= 6.0: bonus_mod = 1
                    
            punteggio_con_bonus = punteggio_modulo + bonus_mod
            if punteggio_con_bonus > miglior_punteggio_totale:
                miglior_punteggio_totale = punteggio_con_bonus
                miglior_modulo = modulo['nome']
                migliori_undici = pd.concat(titolari_modulo)
                dettagli_modificatore = bonus_mod
            
    return miglior_modulo, miglior_punteggio_totale, migliori_undici, dettagli_modificatore

# --- 5. INTERFACCIA VISIVA ---
if not df_giornata.empty:
    lista_giocatori = df_giornata['Nome'].tolist()
    tab1, tab2, tab3 = st.tabs(["👤 Analisi Singola", "⚔️ Ballottaggio", "🏆 Formazione Pro"])

    with tab1:
        st.header("Analisi Predittiva Avanzata")
        g_sel = st.selectbox("Giocatore", lista_giocatori)
        dati_g = df_giornata[df_giornata['Nome'] == g_sel]
        
        xfp, vbase, min_v, max_v = calcola_previsioni(dati_g)
        st.info(f"🏟️ **Contro:** {dati_g['Avversario'].values[0]} | **Difficoltà per il suo ruolo:** {dati_g['Difficolta_Ruolo'].values[0]}/5 | **Min. Giocati scorsi:** {dati_g['Fatica'].values[0]}")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Totale Atteso (xFP)", f"{xfp}")
        c2.metric("Voto Base Puro", f"{vbase}")
        c3.metric("Scen. Peggiore", f"{min_v}")
        c4.metric("Scen. Migliore", f"{max_v}")

    with tab2:
        st.header("Ballottaggio Titolare")
        colA, colB = st.columns(2)
        with colA:
            g_A = st.selectbox("Giocatore A", lista_giocatori, index=0)
            d_A = df_giornata[df_giornata['Nome'] == g_A]
        with colB:
            g_B = st.selectbox("Giocatore B", lista_giocatori, index=min(1, len(lista_giocatori)-1))
            d_B = df_giornata[df_giornata['Nome'] == g_B]

        xfp_A, vbase_A, min_A, max_A = calcola_previsioni(d_A)
        xfp_B, vbase_B, min_B, max_B = calcola_previsioni(d_B)

        st.divider()
        st.success(f"💡 Il Modello consiglia: **{g_A if xfp_A > xfp_B else g_B}**")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[xfp_B, xfp_A], y=[g_B, g_A], mode='markers',
            marker=dict(color=['orange', 'blue'], size=15),
            error_x=dict(type='data', symmetric=False, array=[max_B-xfp_B, max_A-xfp_A], arrayminus=[xfp_B-min_B, xfp_A-min_A], color='gray', thickness=3, width=8)
        ))
        fig.update_layout(xaxis_title="Fantapunti Attesi (con probabilità Bonus inclusa)", height=300)
        st.plotly_chart(fig, use_container_width=True, key="grafico_ballottaggio")
        
    with tab3:
        st.header("La Formazione Perfetta (Ottimizzazione Globale)")
        mod_top, pt_top, top_11, bonus = suggerisci_miglior_formazione(df_giornata)
        
        if top_11.empty:
            st.warning("⚠️ Non hai inserito abbastanza giocatori per completare un modulo.")
        else:
            msg_bonus = f"(di cui +{bonus} da Modificatore chirurgico)" if bonus > 0 else "(Senza modificatore)"
            st.success(f"🔥 Modulo Consigliato: **{mod_top}** | Punti Attesi: **{pt_top:.2f}** {msg_bonus}")
            
            c_P, c_D, c_C, c_A = st.columns(4)
            with c_P:
                st.subheader("Portiere")
                for _, g in top_11[top_11['Ruolo'] == 'P'].iterrows(): st.info(f"🧤 **{g['Nome']}** ({g['xFP']} pt)")
            with c_D:
                st.subheader("Difesa")
                for _, g in top_11[top_11['Ruolo'] == 'D'].iterrows(): st.info(f"🛡️ **{g['Nome']}** ({g['xFP']} pt)")
            with c_C:
                st.subheader("Centrocampo")
                for _, g in top_11[top_11['Ruolo'] == 'C'].iterrows(): st.info(f"👟 **{g['Nome']}** ({g['xFP']} pt)")
            with c_A:
                st.subheader("Attacco")
                for _, g in top_11[top_11['Ruolo'] == 'A'].iterrows(): st.info(f"⚽ **{g['Nome']}** ({g['xFP']} pt)")
