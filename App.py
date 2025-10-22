import streamlit as st
from supabase import create_client
import pandas as pd
import requests
import json
import plotly.express as px
from dotenv import load_dotenv
import os
import subprocess
import matplotlib  # <- Import matplotlib untuk background_gradient

# ==============================================================  
# CONFIG & INITIALIZATION  
# ==============================================================  

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OLLAMA_EXE = os.getenv("OLLAMA_EXE")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3")

# Pastikan Ollama berjalan
def ensure_ollama_running():
    try:
        requests.get("http://localhost:11434")
    except requests.exceptions.ConnectionError:
        st.info("🚀 Starting Ollama service...")
        subprocess.Popen([OLLAMA_EXE, "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

ensure_ollama_running()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="AI Talent Match Intelligence", layout="wide")
st.title("🧠 AI-Powered Talent Match Intelligence Dashboard")

# ==============================================================  
# SIDEBAR INPUTS  
# ==============================================================  

st.sidebar.header("🔧 Input Parameters")

try:
    roles_response = supabase.table("dim_positions").select("name").execute()
    available_roles = sorted([r["name"] for r in roles_response.data if r["name"]])
except Exception:
    available_roles = ["Data Analyst", "HRBP", "Brand Executive", "Sales Supervisor"]

role_name = st.sidebar.selectbox("Select Role Name", available_roles, index=0)
job_level = st.sidebar.selectbox("Job Level", ["Junior", "Middle", "Senior"], index=1)
role_purpose = st.sidebar.text_area(
    "Role Purpose",
    "Analyze and visualize data for business decision-making."
)
benchmark_ids = st.sidebar.text_input("Benchmark Employee IDs (comma-separated)", "312, 335, 175")

# ==============================================================  
# MAIN EXECUTION  
# ==============================================================  

if st.sidebar.button("Run Talent Match"):
    with st.spinner("Running Talent Match..."):
        # Step 1: Record benchmark info
        try:
            supabase.table("talent_benchmarks").insert({
                "role_name": role_name,
                "job_level": job_level,
                "role_purpose": role_purpose,
                "benchmark_ids": benchmark_ids
            }).execute()
            st.success("✅ Job benchmark recorded successfully.")
        except Exception as e:
            st.warning(f"⚠️ Unable to insert benchmark record: {e}")

        # Step 2: Run RPC function
        benchmark_list = [x.strip() for x in benchmark_ids.split(",")]

        try:
            sql_result = supabase.rpc("run_talent_match", {"benchmark_ids": benchmark_list}).execute()

            if not sql_result.data:
                st.error("❌ No data retrieved from Supabase. Check function or parameters.")
                st.stop()

            df = pd.DataFrame(sql_result.data)
            st.success("✅ Talent match computed successfully!")

            # ==============================================================  
            # AI-GENERATED JOB PROFILE (Phi-3 via Ollama local)  
            # ==============================================================  

            st.subheader("🤖 AI-Generated Job Profile")

            prompt_text = f"""
            You are an HR analyst AI. Based on this role info:
            Role: {role_name}
            Level: {job_level}
            Purpose: {role_purpose}
            Benchmarks: {benchmark_ids}

            Generate:
            1. Key job requirements
            2. Role description
            3. Competency highlights
            """

            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": OLLAMA_MODEL, "prompt": prompt_text, "stream": False}
                )
                response.raise_for_status()
                result_text = response.json().get("response", "No response from model.")
                st.markdown(result_text)
            except Exception as e:
                st.warning(f"⚠️ Skipping AI generation: {e}")

            # ==============================================================  
            # DATA CLEANING & FILTERING  
            # ==============================================================  

            df['out_final_match_rate'] = df['out_final_match_rate'].astype(float)
            df['out_tgv_match_rate'] = df['out_tgv_match_rate'].astype(float)
            df['out_tv_match_rate'] = df['out_tv_match_rate'].astype(float)

            df = df[df["out_role"].str.lower() == role_name.lower()]

            if df.empty:
                st.warning(f"⚠️ Tidak ada kandidat dengan role '{role_name}' ditemukan.")
                st.stop()

            # ==============================================================  
            # AGGREGATE FINAL MATCHES  
            # ==============================================================  

            df_final = (
                df.groupby(['out_employee_id', 'out_fullname', 'out_directorate', 'out_role', 'out_grade'])
                .agg({'out_final_match_rate': 'mean'})
                .reset_index()
                .sort_values(by='out_final_match_rate', ascending=False)
            )

            # ==============================================================  
            # DASHBOARD LAYOUT  
            # ==============================================================  

            tab1, tab2, tab3 = st.tabs(["🏅 Final Match", "📈 TGV Breakdown", "🔍 TV Details"])

            # TAB 1
            with tab1:
                st.markdown(f"### 🏅 Ranked Talent List — Role: **{role_name}**")
                st.dataframe(df_final.style.background_gradient(cmap="Greens"))

                fig = px.bar(
                    df_final.head(10),
                    x="out_fullname",
                    y="out_final_match_rate",
                    text="out_final_match_rate",
                    color="out_final_match_rate",
                    color_continuous_scale="Blues",
                    title=f"Top 10 {role_name} Candidates"
                )
                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                fig.update_layout(yaxis_title="Final Match Rate (%)", xaxis_title="Employee")
                st.plotly_chart(fig, use_container_width=True)

            # TAB 2
            with tab2:
                st.markdown("### 📈 TGV Match Breakdown per Employee")
                df_tgv = df[['out_employee_id', 'out_fullname', 'out_tgv_name', 'out_tgv_match_rate']].drop_duplicates()
                st.dataframe(df_tgv.style.background_gradient(cmap="Purples"))
                fig2 = px.bar(
                    df_tgv,
                    x="out_fullname",
                    y="out_tgv_match_rate",
                    color="out_tgv_name",
                    barmode="group",
                    title=f"TGV Match Breakdown for {role_name}"
                )
                st.plotly_chart(fig2, use_container_width=True)

            # TAB 3
            with tab3:
                st.markdown("### 🔍 Detailed TV Comparison")
                df_tv = df[['out_fullname', 'out_tgv_name', 'out_tv_name',
                            'out_user_score', 'out_baseline_score', 'out_tv_match_rate']]
                st.dataframe(df_tv.style.background_gradient(cmap="Oranges"))

            # ==============================================================  
            # INSIGHTS SUMMARY  
            # ==============================================================  

            st.subheader("📊 Insights Summary")
            top_person = df_final.iloc[0]
            top_tgv = (
                df[df['out_fullname'] == top_person['out_fullname']]
                .sort_values('out_tgv_match_rate', ascending=False)
                .iloc[0]
            )

            st.markdown(f"""
            **Top Candidate:** {top_person['out_fullname']}  
            **Directorate:** {top_person['out_directorate']}  
            **Role:** {top_person['out_role']}  
            **Grade:** {top_person['out_grade']}  
            **Final Match Rate:** {top_person['out_final_match_rate']:.2f}%  
            **Strongest TGV:** {top_tgv['out_tgv_name']} ({top_tgv['out_tgv_match_rate']:.2f}%)  
            """)

        except Exception as e:
            st.error(f"❌ Error executing talent match RPC: {e}")