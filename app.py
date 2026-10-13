import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Fantacalcio AI", layout="wide")
st.title("⚽ Fantacalcio: L'Oracolo🔮")

# --- 1. DIZIONARIO DIFFICOLTÀ ---
DIFFICOLTA_SQUADRE = {
    'Inter': 5, 'Juventus': 5, 'Napoli': 5, 'Milan': 4, 'Atalanta': 4, 
    'Roma': 4, 'Lazio': 4, 'Fiorentina': 3, 'Torino': 3, 'Bologna': 3,
    'Udinese': 2, 'Genoa': 2, 'Monza': 2, 'Lecce': 2, 'Empoli': 2,
    'Verona': 2, 'Cagliari': 1, 'Parma': 1, 'Como': 1, 'Venezia': 1
}

# --- MODULI FANTACALCIO VALIDI ---
MODULI_VALIDI = [
    {'nome': '3-4-3', 'P': 1, 'D': 3, 'C': 4, 'A': 3},
    {'nome': '3-5-2', 'P': 1, 'D': 3, 'C': 5, 'A': 2},
    {'nome': '4-3-3', 'P': 1, 'D': 4, 'C': 3, 'A': 3},
    {'nome': '4-4-2', 'P': 1, 'D': 4, 'C': 4, 'A': 2},
    {'nome': '4-5-1', 'P': 1, 'D': 4, 'C': 5, 'A': 1},
    {'nome': '5-3-2', 'P': 1, 'D': 5, 'C': 3, 'A': 2},
    {'nome': '5-4-1', 'P': 1, 'D': 5, 'C': 4, 'A': 1}
]

# --- 2. CARICAMENTO DATI REALI ---
@st.cache_data
def prepara_dati_giornata(giornata_da_giocare):
    # Controllo che i file esistano
    if not os.path.exists('giocatori.csv'):
        st.error("ERRORE: Manca il file 'giocatori.csv' nella cartella!")
        return pd.DataFrame()
        
    giocatori = pd.read_csv('giocatori.csv', sep=';')
    calendario = pd.read_csv('calendario.csv', sep=';')
    
    # 1. Legge il nuovo file orizzontale (matrice)
    voti_orizzontali = pd.read_csv('storico_voti.csv', sep=';')
    
    # 2. Usa 'melt' per trasformare le colonne (G1, G2...) in righe
    voti = voti_orizzontali.melt(id_vars=['Nome'], var_name='Giornata', value_name='Fantavoto')
    
    # 3. Rimuove la 'G' (se presente) e converte in numero
    voti['Giornata'] = voti['Giornata'].astype(str).str.replace('G', '').astype(int)
    
    # NOVITA': Converte 's.v.' o testi in valori nulli (NaN)
    voti['Fantavoto'] = pd.to_numeric(voti['Fantavoto'], errors='coerce')
    
    # 4. Rimuove le righe vuote (es. infortuni, s.v. non numerici o giornate non giocate)
    voti = voti.dropna(subset=['Fantavoto'])

    # Uniamo voti e anagrafica
    df_completo = pd.merge(voti, giocatori, on='Nome', how='left')
    df_completo = df_completo.sort_values(by=['Nome', 'Giornata'])
    
    # Calcoliamo la media mobile delle ultime 3 partite in automatico
    df_completo['Media_Ultime_3'] = df_completo.groupby('Nome')['Fantavoto'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean().fillna(6.0)
    )
    
    partite_oggi = calendario[calendario['Giornata'] == giornata_da_giocare]
    prossima_giornata = []

    for _, row in giocatori.iterrows():
        nome, squadra, ruolo = row['Nome'], row['Squadra'], row['Ruolo']
        
        storico_g = df_completo[df_completo['Nome'] == nome]
        ultima_media = storico_g['Media_Ultime_3'].iloc[-1] if not storico_g.empty else 6.0
        
        match_casa = partite_oggi[partite_oggi['Squadra_Casa'] == squadra]
        match_trasf = partite_oggi[partite_oggi['Squadra_Trasferta'] == squadra]
        
        if not match_casa.empty:
            fattore_campo, avversario = 1, match_casa['Squadra_Trasferta'].values[0]
        elif not match_trasf.empty:
            fattore_campo, avversario = 0, match_trasf['Squadra_Casa'].values[0]
        else:
            continue # Salta se la squadra non gioca
            
        difficolta = DIFFICOLTA_SQUADRE.get(avversario, 3) 
        
        prossima_giornata.append({
            'Nome': nome, 'Ruolo': ruolo, 'Squadra': squadra,
            'Media_Ultime_3': round(ultima_media, 2),
            'Fattore_Campo': fattore_campo, 'Avversario': avversario,
            'Difficolta_Avversario': difficolta
        })

    return pd.DataFrame(prossima_giornata)

# --- 3. MOTORE INFERENZIALE ---
@st.cache_resource
def allena_modello():
    dati_train = pd.DataFrame({
        'Fantavoto': [6.0, 5.0, 9.5, 6.5, 7.0, 6.0, 6.5, 8.0, 5.0, 6.5, 5.5, 7.5, 4.5, 6.0, 5.5, 7.0, 10.5, 4.0], 
        'Media_Ultime_3': [6.5, 6.0, 8.0, 6.5, 7.0, 6.0, 6.5, 7.5, 5.5, 6.8, 6.0, 7.0, 5.0, 6.0, 5.5, 6.5, 8.5, 5.0], 
        'Difficolta_Avversario': [4, 5, 2, 3, 5, 3, 2, 1, 5, 3, 4, 2, 5, 3, 4, 2, 1, 5], 
        'Fattore_Campo': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
        'Ruolo': ['A', 'A', 'A', 'A', 'C', 'C', 'C', 'C', 'D', 'D', 'D', 'D', 'P', 'P', 'P', 'P', 'A', 'P']
    })
    formula = 'Fantavoto ~ Media_Ultime_3 + C(Ruolo) * Difficolta_Avversario + Fattore_Campo'
    return smf.ols(formula=formula, data=dati_train).fit()

giornata_input = st.sidebar.number_input("Imposta la Giornata da giocare", min_value=1, max_value=38, value=3)
df_giornata = prepara_dati_giornata(giornata_input)
modello = allena_modello()

def calcola_previsione(dati_giocatore):
    prev = modello.get_prediction(dati_giocatore).summary_frame(alpha=0.05)
    return prev['mean'].values[0], prev['obs_ci_lower'].values[0], prev['obs_ci_upper'].values[0]

def suggerisci_miglior_formazione(df_giornata, modello):
    df_giornata['xFP'] = modello.predict(df_giornata)
    df_ordinato = df_giornata.sort_values(by=['Ruolo', 'xFP'], ascending=[True, False])
    
    miglior_modulo = None
    miglior_punteggio_totale = 0
    migliori_undici = pd.DataFrame()
    dettagli_modificatore = 0 # Per dirti quanti punti ha preso dal modificatore

    for modulo in MODULI_VALIDI:
        titolari_modulo = []
        punteggio_modulo = 0
        modulo_valido = True
        
        # Dizionario per salvare chi abbiamo scelto, ci serve per il modificatore
        scelti_per_ruolo = {}
        
        for ruolo in ['P', 'D', 'C', 'A']:
            giocatori_disponibili = df_ordinato[df_ordinato['Ruolo'] == ruolo]
            posti_da_titolare = modulo[ruolo]
            
            if len(giocatori_disponibili) < posti_da_titolare:
                modulo_valido = False
                break
                
            scelti = giocatori_disponibili.head(posti_da_titolare)
            titolari_modulo.append(scelti)
            scelti_per_ruolo[ruolo] = scelti
            punteggio_modulo += scelti['xFP'].sum()
            
        if modulo_valido:
            # --- LOGICA MODIFICATORE DIFESA ---
            bonus_mod = 0
            if modulo['D'] >= 4:
                # Prende il voto del Portiere
                voto_p = scelti_per_ruolo['P']['xFP'].values[0]
                # Prende i voti dei migliori 3 difensori schierati
                voti_d = scelti_per_ruolo['D'].head(3)['xFP'].tolist()
                
                media_mod = (voto_p + sum(voti_d)) / 4
                
                # Fasce classiche del modificatore
                if media_mod >= 7.0:
                    bonus_mod = 6
                elif media_mod >= 6.5:
                    bonus_mod = 3
                elif media_mod >= 6.0:
                    bonus_mod = 1
                    
            # Aggiunge il bonus al punteggio di questo modulo
            punteggio_con_bonus = punteggio_modulo + bonus_mod
            
            # Controlla se è il punteggio più alto mai registrato
            if punteggio_con_bonus > miglior_punteggio_totale:
                miglior_punteggio_totale = punteggio_con_bonus
                miglior_modulo = modulo['nome']
                migliori_undici = pd.concat(titolari_modulo)
                dettagli_modificatore = bonus_mod
            
    return miglior_modulo, miglior_punteggio_totale, migliori_undici, dettagli_modificatore

# --- 4. INTERFACCIA VISIVA ---
if not df_giornata.empty:
    lista_giocatori = df_giornata['Nome'].tolist()
    tab1, tab2, tab3 = st.tabs(["👤 Analisi Singola", "⚔️ Ballottaggio", "🏆 Formazione Ottimizzata"])

    with tab1:
        st.header("Analizza un tuo giocatore")
        giocatore_sel = st.selectbox("Giocatore", lista_giocatori)
        dati_g = df_giornata[df_giornata['Nome'] == giocatore_sel]
        
        luogo = "in Casa" if dati_g['Fattore_Campo'].values[0] == 1 else "in Trasferta"
        st.info(f"🏟️ **Partita:** {dati_g['Squadra'].values[0]} contro {dati_g['Avversario'].values[0]} ({luogo}) | **Difficoltà Avversario:** {dati_g['Difficolta_Avversario'].values[0]}/5")

        xfp, min_v, max_v = calcola_previsione(dati_g)
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Fantapoints", f"{xfp:.2f}")
        c2.metric("Scenario Peggiore (95%)", f"{min_v:.2f}")
        c3.metric("Scenario Migliore (95%)", f"{max_v:.2f}")

    with tab2:
        st.header("Ballottaggio Titolare")
        colA, colB = st.columns(2)
        
        with colA:
            g_A = st.selectbox("Giocatore A", lista_giocatori, index=0)
            dati_A = df_giornata[df_giornata['Nome'] == g_A]
            st.caption(f"Contro: {dati_A['Avversario'].values[0]} | Forma: {dati_A['Media_Ultime_3'].values[0]}")
            
        with colB:
            g_B = st.selectbox("Giocatore B", lista_giocatori, index=min(1, len(lista_giocatori)-1))
            dati_B = df_giornata[df_giornata['Nome'] == g_B]
            st.caption(f"Contro: {dati_B['Avversario'].values[0]} | Forma: {dati_B['Media_Ultime_3'].values[0]}")

        # 1. L'algoritmo calcola le previsioni
        xfp_A, min_A, max_A = calcola_previsione(dati_A)
        xfp_B, min_B, max_B = calcola_previsione(dati_B)

        st.divider()
        
        # 2. Mostra il vincitore a schermo!
        migliore = g_A if xfp_A > xfp_B else g_B
        st.success(f"💡 L'algoritmo consiglia di schierare: **{migliore}**")

        # 3. Disegna il grafico
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[xfp_B, xfp_A], y=[g_B, g_A], mode='markers',
            marker=dict(color=['orange', 'blue'], size=15),
            error_x=dict(
                type='data', symmetric=False, 
                array=[max_B-xfp_B, max_A-xfp_A], 
                arrayminus=[xfp_B-min_B, xfp_A-min_A], 
                color='gray', thickness=3, width=8
            )
        ))
        fig.update_layout(xaxis_title="Voto Atteso (xFP) con Rischio (Barra Grigia)", xaxis=dict(range=[2, 11]), height=300)
        st.plotly_chart(fig, use_container_width=True, key="grafico_ballottaggio")
    with tab3:
        st.header("La Formazione Perfetta")
        st.markdown("L'algoritmo ha calcolato il voto atteso di tutta la tua rosa e ha testato tutti i moduli possibili per trovare la combinazione che massimizza il punteggio totale.")
        
        modulo_top, punti_top, top_11, bonus_mod = suggerisci_miglior_formazione(df_giornata, modello)
        
        if top_11.empty:
            st.warning("⚠️ Non hai inserito abbastanza giocatori per completare un modulo valido (es. manca il portiere).")
        else:
            messaggio_bonus = f"(di cui +{bonus_mod} da Modificatore)" if bonus_mod > 0 else "(Senza modificatore)"
            st.success(f"🔥 Modulo Consigliato: **{modulo_top}** | Punti Attesi: **{punti_top:.2f}** {messaggio_bonus}")
            
            # Creiamo una visualizzazione a colonne per i reparti
            col_P, col_D, col_C, col_A = st.columns(4)
            
            with col_P:
                st.subheader("Portiere")
                for _, g in top_11[top_11['Ruolo'] == 'P'].iterrows():
                    st.info(f"🧤 **{g['Nome']}** ({g['xFP']:.1f} pt)")
            
            with col_D:
                st.subheader("Difesa")
                for _, g in top_11[top_11['Ruolo'] == 'D'].iterrows():
                    st.info(f"🛡️ **{g['Nome']}** ({g['xFP']:.1f} pt)")
                    
            with col_C:
                st.subheader("Centrocampo")
                for _, g in top_11[top_11['Ruolo'] == 'C'].iterrows():
                    st.info(f"👟 **{g['Nome']}** ({g['xFP']:.1f} pt)")
                    
            with col_A:
                st.subheader("Attacco")
                for _, g in top_11[top_11['Ruolo'] == 'A'].iterrows():
                    st.info(f"⚽ **{g['Nome']}** ({g['xFP']:.1f} pt)")
            
            st.divider()
            st.caption("Nota: I punti attesi sono voti base teorici matematici.")
else:
    st.warning("Assicurati di aver creato i 3 file CSV nella cartella e che i nomi delle colonne siano corretti.")
