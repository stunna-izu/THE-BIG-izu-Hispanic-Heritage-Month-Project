import io
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sympy as sp

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer 2.0", page_icon="📈", layout="wide")

# ----------------------------- Utilities -----------------------------
CONTEXTS = {
    "Business": ("revenue", "time", "USD", "year", "Revenue over time; derivatives describe the rate and acceleration of revenue change."),
    "Engineering": ("position", "time", "meters", "seconds", "Position, velocity, and acceleration describe motion."),
    "Medical": ("concentration", "time", "mg/L", "hours", "A concentration curve can describe how a measured quantity changes over time; it is illustrative, not medical advice."),
    "Sports science": ("performance", "time", "score units", "seconds", "The derivative describes the rate of performance change in the chosen model."),
    "Biology": ("population", "time", "organisms", "days", "The model describes population size and its rate of change."),
    "Bioengineering": ("signal", "time", "signal units", "seconds", "The model describes a biological or engineered signal and its changing rate."),
    "Economic forecasting": ("index", "time", "index points", "years", "The derivative describes the index's rate of change; the model is not a guarantee of future outcomes."),
    "Weather": ("temperature", "time", "°C", "hours", "The derivative describes how quickly temperature changes."),
    "Custom": ("output", "input", "output units", "input units", "A user-defined relationship.")
}
SUGGESTED = {
    "Business": "0.4*x**2 + 3*x + 20",
    "Engineering": "0.5*x**2 - 4*x + 2",
    "Medical": "8*exp(-0.25*x) + 1",
    "Sports science": "-0.1*(x-5)**2 + 10",
    "Biology": "2*exp(0.15*x)",
    "Bioengineering": "sin(x) + 0.2*x",
    "Economic forecasting": "0.03*x**2 + 0.5*x + 100",
    "Weather": "5*sin(2*pi*x/24) + 15",
    "Custom": "x**2 - 4*x + 3"
}

def safe_expr(raw, x):
    """Parse familiar math notation with a restricted SymPy namespace."""
    raw = raw.strip().replace("^", "**")
    if not raw:
        raise ValueError("Enter a function, such as x**2 - 4*x + 3.")
    allowed = {"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
               "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
               "exp": sp.exp, "log": sp.log, "ln": sp.log, "sqrt": sp.sqrt,
               "abs": sp.Abs, "pi": sp.pi, "E": sp.E, "cosh": sp.cosh,
               "sinh": sp.sinh, "tan": sp.tan}
    try:
        expr = sp.sympify(raw, locals=allowed, evaluate=True)
    except Exception as e:
        raise ValueError(f"Could not parse that function. Check parentheses, operators, and function names. Details: {e}")
    if expr.free_symbols - {x}:
        raise ValueError("Use x as the independent variable; unknown symbols were found.")
    return expr

def numeric_fn(expr, x):
    f = sp.lambdify(x, expr, modules=["numpy"])
    def wrapped(vals):
        arr = np.asarray(vals, dtype=float)
        with np.errstate(all="ignore"):
            out = f(arr)
        out = np.asarray(out)
        if out.ndim == 0:
            out = np.full(arr.shape, out.item())
        try:
            out = np.broadcast_to(out, arr.shape).astype(float)
        except Exception:
            out = np.asarray([float(f(float(v))) for v in arr])
        out[~np.isfinite(out)] = np.nan
        return out
    return wrapped

def fmt(v, digits=5):
    try:
        if not np.isfinite(float(v)): return "undefined"
        return f"{float(v):.{digits}g}"
    except Exception:
        return str(v)

def derivative_analysis(expr, x, lo, hi):
    d1, d2 = sp.diff(expr, x), sp.diff(expr, x, 2)
    critical = []
    inflections = []
    notes = []
    try:
        roots = sp.solveset(sp.Eq(d1, 0), x, domain=sp.Interval(lo, hi))
        if isinstance(roots, sp.FiniteSet):
            critical = sorted([float(sp.N(r)) for r in roots if abs(float(sp.im(sp.N(r)))) < 1e-8])
        else:
            notes.append("Some critical points may not have a closed-form solution; numerical scan is used for graphing.")
    except Exception:
        notes.append("Critical-point solving was not available symbolically for this function.")
    # Numerical fallback/additional roots for critical points and second derivative
    grid = np.linspace(lo, hi, 1200)
    for deriv, dest in [(d1, critical), (d2, inflections)]:
        try:
            fn = numeric_fn(deriv, x)
            yy = fn(grid)
            for i in range(len(grid)-1):
                if np.isfinite(yy[i]) and np.isfinite(yy[i+1]) and yy[i] * yy[i+1] < 0:
                    a,b=grid[i],grid[i+1]
                    for _ in range(35):
                        m=(a+b)/2
                        ym=fn(np.array([m]))[0]
                        if not np.isfinite(ym): break
                        if yy[i]*ym <= 0: b=m
                        else: a=m
                    dest.append((a+b)/2)
        except Exception:
            pass
    critical = sorted(set(round(v, 7) for v in critical if lo-1e-7 <= v <= hi+1e-7))
    inflections = sorted(set(round(v, 7) for v in inflections if lo-1e-7 <= v <= hi+1e-7))
    f, f1, f2 = numeric_fn(expr,x), numeric_fn(d1,x), numeric_fn(d2,x)
    classified=[]
    for c in critical:
        eps=max((hi-lo)*1e-5, 1e-5)
        left,right=c-eps,c+eps
        dl,dr=f1(np.array([left,right]))
        if np.isfinite(dl) and np.isfinite(dr) and dl>0 and dr<0: kind="local maximum"
        elif np.isfinite(dl) and np.isfinite(dr) and dl<0 and dr>0: kind="local minimum"
        else: kind="stationary point (not classified as an extremum)"
        classified.append((c, float(f(np.array([c]))[0]), kind))
    verified=[]
    for c in inflections:
        eps=max((hi-lo)*1e-5,1e-5)
        vals=f2(np.array([c-eps,c+eps]))
        if np.isfinite(vals).all() and vals[0]*vals[1] < 0:
            verified.append((c, float(f(np.array([c]))[0])))
    return d1,d2,critical,classified,inflections,verified,notes

def poly_fit(df, degree, base_year):
    if df["year"].nunique() < degree+1:
        raise ValueError(f"Degree {degree} requires at least {degree+1} distinct years.")
    t = df["year"].astype(float).to_numpy() - float(base_year)
    y = df["value"].astype(float).to_numpy()
    coeff = np.polyfit(t, y, degree)
    pred = np.polyval(coeff, t)
    residuals = y-pred
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y-y.mean())**2))
    r2 = float(1-ss_res/ss_tot) if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(residuals)))
    return coeff, pred, residuals, r2, mae

def polynomial_text(coeff, base_year):
    terms=[]
    degree=len(coeff)-1
    for i,c in enumerate(coeff):
        p=degree-i
        if abs(c)<1e-14: continue
        if p==0: terms.append(f"({c:.10g})")
        elif p==1: terms.append(f"({c:.10g})·t")
        else: terms.append(f"({c:.10g})·t^{p}")
    return "y = " + (" + ".join(terms) if terms else "0") + f", where t = year − {base_year}"

def make_pdf(summary, chart_bytes, preview, results):
    # ReportLab is a required dependency; create a real PDF with text and chart.
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=letter,rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36)
    styles=getSampleStyleSheet()
    story=[Paragraph("Hispanic Heritage Historical Data Explorer 2.0",styles["Title"]),Spacer(1,8)]
    for line in summary:
        story.append(Paragraph(str(line).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"),styles["BodyText"]))
        story.append(Spacer(1,4))
    if chart_bytes:
        try:
            story.extend([Spacer(1,8),Image(io.BytesIO(chart_bytes),width=6.5*inch,height=3.4*inch)])
        except Exception:
            pass
    for title, data in [("Data preview",preview),("Calculated-year results",results)]:
        story.extend([Spacer(1,8),Paragraph(title,styles["Heading2"])])
        if data is not None and len(data):
            vals=[list(map(str,data.columns))]+data.astype(str).head(18).values.tolist()
            table=Table(vals,repeatRows=1)
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
            story.append(table)
        else: story.append(Paragraph("No rows available.",styles["BodyText"]))
    doc.build(story)
    return buf.getvalue()

# ----------------------------- UI -----------------------------
st.title("🌎 Hispanic Heritage Historical Data Explorer 2.0")
st.caption("Explore historical indicators, fit trends, and learn how functions change. Educational models are not guarantees of future outcomes.")
tab_data, tab_func, tab_guide = st.tabs(["Historical Data Explorer", "Function & Calculus Lab", "Run & Deploy"])

with tab_data:
    st.subheader("Historical data and polynomial model")
    c1,c2,c3,c4=st.columns(4)
    country=c1.selectbox("Country",["Cuba","Mexico","Puerto Rico","Dominican Republic","Colombia","Argentina","Chile","Costa Rica","Spain","Custom / uploaded data"])
    indicator=c2.selectbox("Indicator",["Population (illustrative demo data)","GDP index (illustrative demo data)","Life expectancy (illustrative demo data)","Custom indicator"])
    degree=c3.selectbox("Polynomial degree",list(range(1,7)),index=2)
    base_year=c4.number_input("Base year for centered time",min_value=1800,max_value=2100,value=2000,step=1)
    st.info("The built-in series are clearly labeled illustrative demo data, not sourced historical statistics. Upload a sourced CSV to analyze real observations.")
    upload=st.file_uploader("Upload CSV with columns year,value (optional)",type=["csv"],key="hist_upload")
    if upload is not None:
        try:
            raw=pd.read_csv(upload)
            if not {"year","value"}.issubset({str(c).strip().lower() for c in raw.columns}):
                # Normalize case/whitespace
                raw.columns=[str(c).strip().lower() for c in raw.columns]
            else: raw.columns=[str(c).strip().lower() for c in raw.columns]
            if not {"year","value"}.issubset(raw.columns): raise ValueError("CSV must include year and value columns.")
            data=raw[["year","value"]].copy()
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            data=None
    else:
        years=np.arange(1990,2026,5)
        seed=sum(ord(ch) for ch in country+indicator)
        rng=np.random.default_rng(seed)
        if indicator.startswith("Population"):
            values=1.5e6 + (years-1990)*12000 + 80000*np.sin((years-1990)/8) + rng.normal(0,18000,len(years))
            unit="people"
        elif indicator.startswith("GDP"):
            values=45 + (years-1990)*1.9 + 7*np.sin((years-1990)/6) + rng.normal(0,2,len(years))
            unit="index points"
        elif indicator.startswith("Life"):
            values=66 + (years-1990)*0.25 + 1.5*np.sin((years-1990)/7) + rng.normal(0,.4,len(years))
            unit="years"
        else:
            values=50 + (years-1990)*.8 + 5*np.sin((years-1990)/5) + rng.normal(0,1,len(years))
            unit="custom units"
        data=pd.DataFrame({"year":years,"value":np.round(values,3)})
    if data is not None:
        data["year"]=pd.to_numeric(data["year"],errors="coerce")
        data["value"]=pd.to_numeric(data["value"],errors="coerce")
        data=data.dropna(subset=["year","value"])
        data=data[(data["year"]>=1)&(data["year"]<=9999)].copy()
        data["year"]=data["year"].astype(int)
        data=data.sort_values("year").reset_index(drop=True)
        st.markdown("**Working data** — edit values directly; duplicate years are allowed as observations but distinct years determine fit eligibility.")
        edited=st.data_editor(data,use_container_width=True,num_rows="dynamic",key="data_editor")
        try:
            edited["year"]=pd.to_numeric(edited["year"],errors="raise").astype(int)
            edited["value"]=pd.to_numeric(edited["value"],errors="raise").astype(float)
            edited=edited.dropna(subset=["year","value"]).sort_values("year").reset_index(drop=True)
            if len(edited)<degree+1 or edited.year.nunique()<degree+1:
                st.warning(f"Need at least {degree+1} rows and distinct years for degree {degree}.")
                fit_ok=False
            else: fit_ok=True
        except Exception as e:
            st.error(f"Invalid working data: {e}")
            fit_ok=False
        unit_map={"Population (illustrative demo data)":"people","GDP index (illustrative demo data)":"index points","Life expectancy (illustrative demo data)":"years","Custom indicator":"custom units"}
        unit=unit_map.get(indicator,"custom units")
        results_df=pd.DataFrame()
        coeff=None
        if fit_ok:
            try:
                coeff,pred,residuals,r2,mae=poly_fit(edited,degree,base_year)
                fit_start,fit_end=int(edited.year.min()),int(edited.year.max())
                summary=[f"Country/context: {country}",f"Indicator: {indicator}",f"Units: {unit}",f"Base year: {base_year}",f"Degree: {degree}",f"Equation: {polynomial_text(coeff,base_year)}",f"In-sample R²: {r2:.6g}",f"In-sample MAE: {mae:.6g} {unit}",f"Observation count: {len(edited)}",f"Historical range: {fit_start}–{fit_end}",f"Observed mean: {edited.value.mean():.6g} {unit}"]
                a,b,c,d,e=st.columns(5)
                a.metric("In-sample R²",fmt(r2)); b.metric("In-sample MAE",f"{fmt(mae)} {unit}"); c.metric("Observations",len(edited)); d.metric("Historical range",f"{fit_start}–{fit_end}"); e.metric("Observed mean",fmt(edited.value.mean()))
                st.markdown("**Fitted equation**")
                st.code(polynomial_text(coeff,base_year))
                st.caption("Metrics above describe the fitted observations (in-sample); they do not measure guaranteed future accuracy.")
                edited["fitted_value"]=pred
                edited["residual"]=residuals
                with st.expander("Residuals and fitted values at observed years"):
                    st.dataframe(edited,use_container_width=True)
                # Chronological holdout when at least 8 distinct years and enough train points
                distinct=np.sort(data.year.unique())
                if len(distinct)>=8 and len(edited)>=degree+3:
                    ntest=max(1,int(round(len(distinct)*.2)))
                    cutoff=distinct[-ntest]
                    train=edited[edited.year<cutoff]
                    test=edited[edited.year>=cutoff]
                    if train.year.nunique()>=degree+1 and len(test):
                        try:
                            hc,_,_,_,_=poly_fit(train,degree,base_year)
                            hp=np.polyval(hc,test.year.to_numpy()-base_year)
                            hmae=float(np.mean(np.abs(test.value.to_numpy()-hp)))
                            st.write(f"Chronological holdout MAE: **{hmae:.6g} {unit}** (separate from in-sample metrics; train years before {cutoff}).")
                        except Exception:
                            st.caption("Chronological holdout score unavailable for this data/degree.")
                # Single year calculator
                st.markdown("### Calculate a year")
                q1,q2=st.columns([1,1])
                one_year=q1.number_input("Year (1950–2100)",min_value=1950,max_value=2100,value=2025,step=1,key="one_year")
                if q2.button("Calculate Value",type="primary"):
                    val=float(np.polyval(coeff,one_year-base_year))
                    st.session_state["single_result"]={"year":int(one_year),"value":val,"signature":(country,indicator,degree,base_year,tuple(edited.year),tuple(edited.value))}
                sig=(country,indicator,degree,base_year,tuple(edited.year),tuple(edited.value))
                if st.session_state.get("single_result",{}).get("signature")==sig:
                    sr=st.session_state["single_result"]
                    classification="within observed historical range" if fit_start<=sr["year"]<=fit_end else "outside observed historical range (extrapolation)"
                    st.success(f"Year {sr['year']}: **{sr['value']:.8g} {unit}** — {classification}.")
                elif "single_result" in st.session_state:
                    st.session_state.pop("single_result",None)
                # Batch years
                st.markdown("### Batch predictions")
                batch_mode=st.radio("Choose batch input",["Year range","Comma-separated custom years"],horizontal=True)
                years_requested=[]
                if batch_mode=="Year range":
                    bc1,bc2,bc3=st.columns(3)
                    yr_start=bc1.number_input("Start year",1950,2100,2020,1,key="batch_start")
                    yr_end=bc2.number_input("End year",1950,2100,2030,1,key="batch_end")
                    step=bc3.selectbox("Step (1–10)",list(range(1,11)),index=0)
                    if yr_end<yr_start: st.warning("End year must be greater than or equal to start year.")
                    else: years_requested=list(range(int(yr_start),int(yr_end)+1,int(step)))
                else:
                    custom=st.text_input("Years, separated by commas",value="2000, 2025, 2050",key="custom_batch")
                    try:
                        tokens=[v.strip() for v in custom.split(",") if v.strip()]
                        if not tokens: raise ValueError("Enter at least one year.")
                        years_requested=[]
                        for tok in tokens:
                            if not re.fullmatch(r"\d{4}",tok): raise ValueError(f"'{tok}' is not a valid four-digit year.")
                            v=int(tok)
                            if not 1950<=v<=2100: raise ValueError(f"Year {v} is outside 1950–2100.")
                            years_requested.append(v)
                    except Exception as ex:
                        st.error(str(ex)); years_requested=[]
                if st.button("Calculate Batch",key="batch_button"):
                    if years_requested:
                        vals=np.polyval(coeff,np.asarray(years_requested)-base_year)
                        results_df=pd.DataFrame({"year":years_requested,"estimate":vals,"units":unit,
                          "classification":["within observed historical range" if fit_start<=y<=fit_end else "outside observed historical range (extrapolation)" for y in years_requested]})
                        st.session_state["batch_result"]={"data":results_df,"signature":sig}
                br=st.session_state.get("batch_result")
                if br and br["signature"]==sig:
                    results_df=br["data"]
                    st.dataframe(results_df,use_container_width=True)
                elif br:
                    st.session_state.pop("batch_result",None)
                # chart
                st.markdown("### Plot and compare")
                sample_step=st.slider("Sample increment for model markers/table",1,10,2)
                curve_years=np.linspace(fit_start, max(fit_end,2100), 500)
                curve_vals=np.polyval(coeff,curve_years-base_year)
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=edited.year,y=edited.value,mode="markers",name="Observed data",marker=dict(size=9)))
                fig.add_trace(go.Scatter(x=curve_years,y=curve_vals,mode="lines",name="Fitted polynomial (smooth curve)"))
                sampled=curve_years[::max(1,sample_step*10)]
                fig.add_trace(go.Scatter(x=sampled,y=np.polyval(coeff,sampled-base_year),mode="markers",name=f"Model markers (increment {sample_step})"))
                fig.add_vline(x=fit_start,line_dash="dash",annotation_text="Historical start")
                fig.add_vline(x=fit_end,line_dash="dash",annotation_text="Historical end")
                if fit_end<2100: fig.add_vrect(x0=fit_end,x1=2100,fillcolor="gray",opacity=.08,line_width=0,annotation_text="Extrapolation")
                if len(results_df):
                    fig.add_trace(go.Scatter(x=results_df.year,y=results_df.estimate,mode="markers",name="Batch estimates",marker=dict(symbol="diamond",size=11)))
                if st.session_state.get("single_result",{}).get("signature")==sig:
                    sr=st.session_state["single_result"]
                    fig.add_trace(go.Scatter(x=[sr["year"]],y=[sr["value"]],mode="markers",name="Single-year estimate",marker=dict(symbol="star",size=14)))
                fig.update_layout(xaxis_title="Year",yaxis_title=f"Value ({unit})",legend_title="Series",height=480)
                st.plotly_chart(fig,use_container_width=True)
                st.caption("Polynomial extrapolations can become implausible quickly. Estimates outside the observed range are not validated prediction intervals; residuals and MAE only summarize fit error.")
                # exports
                st.markdown("### Export current selection")
                working=edited.copy()
                if len(results_df)==0: results_df=pd.DataFrame(columns=["year","estimate","units","classification"])
                csv_bytes=working.to_csv(index=False).encode("utf-8")
                st.download_button("Download working data CSV",csv_bytes,file_name="historical_working_data.csv",mime="text/csv")
                xbuf=io.BytesIO()
                with pd.ExcelWriter(xbuf,engine="openpyxl") as writer:
                    pd.DataFrame({"item":["country","indicator","unit","base_year","degree","equation","in_sample_r_squared","in_sample_mae","observation_count","historical_range","mean"],
                                  "value":[country,indicator,unit,base_year,degree,polynomial_text(coeff,base_year),r2,mae,len(edited),f"{fit_start}-{fit_end}",edited.value.mean()]}).to_excel(writer,index=False,sheet_name="Summary")
                    working.to_excel(writer,index=False,sheet_name="Working Data")
                    results_df.to_excel(writer,index=False,sheet_name="Calculated Years")
                st.download_button("Download Excel workbook (.xlsx)",xbuf.getvalue(),file_name="historical_explorer.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                try: chart_png=fig.to_image(format="png",width=1100,height=650,scale=1)
                except Exception: chart_png=None
                pdf=make_pdf(summary,chart_png,working,results_df)
                st.download_button("Download PDF report",pdf,file_name="historical_explorer_report.pdf",mime="application/pdf")
                # Country comparison (same indicator)
                st.markdown("### Compare countries")
                compare=st.multiselect("Countries to compare using the same indicator",["Cuba","Mexico","Puerto Rico","Dominican Republic","Colombia","Argentina","Chile","Costa Rica","Spain"],default=["Cuba","Mexico"])
                if compare:
                    compfig=go.Figure()
                    for cn in compare:
                        yrs=np.arange(1990,2026,5)
                        rng=np.random.default_rng(sum(ord(ch) for ch in cn+indicator))
                        if indicator.startswith("Population"): vals=1.5e6+(yrs-1990)*(9000+sum(map(ord,cn))%500)+70000*np.sin((yrs-1990)/8)+rng.normal(0,15000,len(yrs))
                        elif indicator.startswith("GDP"): vals=40+(yrs-1990)*(1.3+sum(map(ord,cn))%10/10)+6*np.sin((yrs-1990)/6)+rng.normal(0,2,len(yrs))
                        else: vals=65+(yrs-1990)*.22+1.2*np.sin((yrs-1990)/7)+rng.normal(0,.5,len(yrs))
                        compfig.add_trace(go.Scatter(x=yrs,y=vals,mode="lines+markers",name=cn))
                    compfig.update_layout(xaxis_title="Year",yaxis_title=f"{indicator} ({unit})",height=400)
                    st.plotly_chart(compfig,use_container_width=True)
                    st.caption("Comparison series are illustrative and may have different coverage in real datasets. Upload comparable, sourced data for defensible country comparisons.")
            except Exception as e:
                st.error(f"Model calculation failed: {e}")
        else:
            st.caption("Fix the working data or choose a lower polynomial degree to enable fitting.")

with tab_func:
    st.subheader("Function & Calculus Lab")
    left,right=st.columns([1,2])
    with left:
        level=st.selectbox("Course mode",["Honors Algebra 2","Precalculus","Calculus"])
        context=st.selectbox("Context",list(CONTEXTS.keys()))
        default_dep,default_ind,default_unit,default_ind_unit,default_desc=CONTEXTS[context]
        desc=st.text_area("Context description",default_desc)
        dep=st.text_input("Dependent variable",default_dep)
        indep=st.text_input("Independent variable",default_ind)
        yunit=st.text_input("Dependent units",default_unit)
        xunit=st.text_input("Independent units",default_ind_unit)
        fn_mode=st.radio("Function source",["Suggested function","Build by parameters","Enter my own","Use historical curve of best fit"],key="fn_mode")
        if fn_mode=="Suggested function":
            func_text=SUGGESTED[context]
            st.code(f"f(x) = {func_text}")
        elif fn_mode=="Build by parameters":
            family=st.selectbox("Function family",["Polynomial (quadratic)","Exponential","Logarithmic","Trigonometric","Rational"])
            if family=="Polynomial (quadratic)":
                a=st.number_input("a",value=1.0); b=st.number_input("b",value=-4.0); c=st.number_input("c",value=3.0)
                func_text=f"({a})*x**2 + ({b})*x + ({c})"
            elif family=="Exponential":
                a=st.number_input("Amplitude a",value=2.0); b=st.number_input("Rate b",value=0.2); c=st.number_input("Offset c",value=1.0)
                func_text=f"({a})*exp(({b})*x)+({c})"
            elif family=="Logarithmic":
                a=st.number_input("Scale a",value=2.0); b=st.number_input("Shift b",value=1.0)
                func_text=f"({a})*log(x+({b}))"
            elif family=="Trigonometric":
                a=st.number_input("Amplitude a",value=2.0); b=st.number_input("Frequency b",value=1.0); c=st.number_input("Vertical shift c",value=0.0)
                func_text=f"({a})*sin(({b})*x)+({c})"
            else:
                a=st.number_input("Numerator scale a",value=1.0); b=st.number_input("Horizontal shift b",value=1.0); c=st.number_input("Vertical shift c",value=0.0)
                func_text=f"({a})/(x-({b}))+({c})"
            st.code(f"f(x) = {func_text}")
        elif fn_mode=="Enter my own":
            func_text=st.text_input("Function f(x)",value="x**2 - 4*x + 3",help="Use x, + - * / **, parentheses, sin, cos, tan, exp, log, sqrt, and pi. You may type ^ for powers.")
        else:
            st.info("Switch to the Historical Data Explorer tab, fit a model, then paste its equation in centered-time form below if you want to reuse it. Automatic cross-tab transfer is intentionally avoided.")
            func_text=st.text_input("Function in x",value="0.03*x**2 + 0.5*x + 100")
        st.markdown("**Analysis interval**")
        lo=st.number_input("Left endpoint",value=-2.0)
        hi=st.number_input("Right endpoint",value=6.0)
        x0=st.slider("Current input x",min_value=float(min(lo,hi)),max_value=float(max(lo,hi)),value=float(min(max(1.0,min(lo,hi)),max(lo,hi))),step=0.1)
        ac1,ac2=st.columns(2)
        aroc_left=ac1.number_input("Average-rate left endpoint",value=float(lo))
        aroc_right=ac2.number_input("Average-rate right endpoint",value=float(hi))
        show_f=st.checkbox("Show f(x)",True)
        show_d1=st.checkbox("Show f′(x)",True)
        show_d2=st.checkbox("Show f″(x)",True)
        play=st.toggle("Auto playback (animate x)",value=False)
        speed=st.select_slider("Playback speed",options=["Slow","Normal","Fast"],value="Normal")
        st.caption("Playback uses Streamlit reruns; move the slider to explore the point. Speed is a classroom control label and does not create a background timer.")
    with right:
        if hi<=lo:
            st.error("The analysis interval must have right endpoint greater than left endpoint.")
        else:
            x=sp.symbols("x", real=True)
            try:
                expr=safe_expr(func_text,x)
                d1=sp.diff(expr,x); d2=sp.diff(expr,x,2)
                st.latex(r"f(x) = "+sp.latex(expr))
                st.latex(r"f'(x) = "+sp.latex(d1))
                st.latex(r"f''(x) = "+sp.latex(d2))
                f=numeric_fn(expr,x); f1=numeric_fn(d1,x); f2=numeric_fn(d2,x)
                grid=np.linspace(float(lo),float(hi),900)
                yy=f(grid); y1=f1(grid); y2=f2(grid)
                fig=go.Figure()
                if show_f: fig.add_trace(go.Scatter(x=grid,y=yy,mode="lines",name="f(x)"))
                if show_d1: fig.add_trace(go.Scatter(x=grid,y=y1,mode="lines",name="f′(x)"))
                if show_d2: fig.add_trace(go.Scatter(x=grid,y=y2,mode="lines",name="f″(x)"))
                fx=float(f(np.array([x0]))[0]); slope=float(f1(np.array([x0]))[0])
                if show_f and np.isfinite(fx):
                    fig.add_trace(go.Scatter(x=[x0],y=[fx],mode="markers",name="Moving point",marker=dict(size=12,symbol="circle")))
                    if np.isfinite(slope):
                        tangent=slope*(grid-x0)+fx
                        fig.add_trace(go.Scatter(x=grid,y=tangent,mode="lines",name="Tangent at current x",line=dict(dash="dash")))
                if np.isfinite(aroc_left) and np.isfinite(aroc_right) and aroc_left!=aroc_right:
                    va=float(f(np.array([aroc_left]))[0]); vb=float(f(np.array([aroc_right]))[0])
                    if np.isfinite(va) and np.isfinite(vb):
                        m=(vb-va)/(aroc_right-aroc_left)
                        sec=m*(grid-aroc_left)+va
                        fig.add_trace(go.Scatter(x=grid,y=sec,mode="lines",name="Secant line",line=dict(dash="dot")))
                fig.add_hline(y=0,line_color="gray",line_width=1)
                fig.add_vline(x=0,line_color="gray",line_width=1)
                fig.update_layout(xaxis_title=f"{indep} ({xunit})",yaxis_title=f"{dep} ({yunit})",height=560,legend_title="Curves")
                st.plotly_chart(fig,use_container_width=True)
                m1,m2,m3=st.columns(3)
                m1.metric(f"{dep} f({x0})",fmt(fx))
                m2.metric("First derivative",fmt(slope)+ (f" {yunit}/{xunit}" if xunit and yunit else ""))
                m3.metric("Second derivative",fmt(float(f2(np.array([x0]))[0]))+(f" {yunit}/{xunit}²" if xunit and yunit else ""))
                if aroc_left!=aroc_right:
                    va=float(f(np.array([aroc_left]))[0]); vb=float(f(np.array([aroc_right]))[0])
                    if np.isfinite(va) and np.isfinite(vb):
                        avg=(vb-va)/(aroc_right-aroc_left)
                        st.write(f"**Average rate of change:** ({fmt(vb)} − {fmt(va)}) / ({fmt(aroc_right)} − {fmt(aroc_left)}) = **{fmt(avg)} {yunit}/{xunit}**.")
                    else: st.warning("Average rate of change is undefined because the function is undefined at an interval endpoint.")
                else: st.warning("Choose different endpoints to calculate average rate of change.")
                st.write(f"**Context interpretation:** {desc} At x = {fmt(x0)}, {dep} is {fmt(fx)} {yunit}. The first derivative is the instantaneous rate of change ({yunit} per {xunit}); the second derivative describes how that rate changes ({yunit} per {xunit}²).")
                if context=="Engineering":
                    st.info("For a position function, f′(x) is velocity and f″(x) is acceleration, when x represents time.")
                if context in ["Business","Economic forecasting"]:
                    st.info("A positive first derivative means the modeled indicator is increasing; a negative one means it is decreasing. The second derivative indicates whether that rate is increasing or decreasing.")
                if level=="Honors Algebra 2":
                    st.markdown("#### Algebra 2 view")
                    finite=yy[np.isfinite(yy)]
                    if len(finite): st.write(f"**Domain on displayed interval:** x ∈ [{lo}, {hi}], excluding any undefined points. **Displayed range (numerical):** approximately [{fmt(np.nanmin(yy))}, {fmt(np.nanmax(yy))}].")
                    st.write("Read the graph's overall trend and compare average rates of change over different intervals. End behavior depends on the function family and leading terms.")
                elif level=="Precalculus":
                    st.markdown("#### Precalculus view")
                    st.write(f"**Degree / structure:** {sp.Poly(expr,x).degree() if expr.is_polynomial(x) else 'non-polynomial'}; coefficients affect scale, shifts, roots, and shape.")
                    st.write("Turning points are candidate locations where the rate changes sign. Concavity describes whether slopes are increasing or decreasing; compare first-derivative values across intervals.")
                    st.write(f"**Numerical average rate on selected interval:** {fmt((float(f(np.array([aroc_right]))[0])-float(f(np.array([aroc_left]))[0]))/(aroc_right-aroc_left)) if aroc_right!=aroc_left and np.isfinite(f(np.array([aroc_left,aroc_right]))).all() else 'undefined'}.")
                else:
                    st.markdown("#### Calculus analysis")
                    d1,d2,critical,classified,inflections,verified,notes=derivative_analysis(expr,x,float(lo),float(hi))
                    if classified:
                        st.write("**Critical points and derivative-test classifications**")
                        st.dataframe(pd.DataFrame([{"x":c,"f(x)":v,"classification":kind} for c,v,kind in classified]),use_container_width=True)
                    else: st.write("No stationary critical points were found numerically/symbolically in the selected interval.")
                    # Sign intervals sampled between critical points
                    cuts=[float(lo)]+critical+[float(hi)]
                    intervals=[]
                    for aa,bb in zip(cuts[:-1],cuts[1:]):
                        mid=(aa+bb)/2; val=float(f1(np.array([mid]))[0])
                        if np.isfinite(val): intervals.append({"interval":f"({fmt(aa)}, {fmt(bb)})","behavior":"increasing" if val>1e-8 else ("decreasing" if val < -1e-8 else "approximately stationary")})
                    if intervals:
                        st.write("**Increasing/decreasing intervals (numerical sign sampling)**")
                        st.dataframe(pd.DataFrame(intervals),use_container_width=True)
                    if verified:
                        st.write("**Verified inflection points (concavity changes numerically)**")
                        st.dataframe(pd.DataFrame([{"x":c,"f(x)":v} for c,v in verified]),use_container_width=True)
                    elif inflections:
                        st.write("Second-derivative zero/candidate locations found, but no concavity change was verified in the interval.")
                    else: st.write("No verified inflection points found in the selected interval.")
                    # absolute extrema including endpoints if valid
                    candidates=[float(lo),float(hi)]+critical
                    vals=[]
                    for c in candidates:
                        v=float(f(np.array([c]))[0])
                        if np.isfinite(v): vals.append((c,v))
                    if vals:
                        mn=min(vals,key=lambda q:q[1]); mx=max(vals,key=lambda q:q[1])
                        st.write(f"**Absolute extrema on the selected closed interval (among evaluated candidates):** minimum at x={fmt(mn[0])}, f(x)={fmt(mn[1])}; maximum at x={fmt(mx[0])}, f(x)={fmt(mx[1])}.")
                    for n in notes: st.caption(n)
                    st.caption("Symbolic results are exact when displayed as symbolic expressions; plotted roots, sign checks, extrema, and inflection verification are numerical approximations. Discontinuities can make a closed-interval absolute-extrema conclusion inapplicable.")
                # domain/discontinuity notes
                den=sp.denom(sp.together(expr))
                if den!=1:
                    st.write(f"**Potential discontinuities:** denominator is {sp.latex(den)}; solve where it equals zero and exclude those x-values from the domain.")
                if expr.has(sp.log):
                    st.write("**Logarithm restriction:** the logarithm argument must be positive.")
                if expr.has(sp.sqrt):
                    st.write("**Square-root restriction:** real-valued square roots require a nonnegative radicand.")
                # simple residual / implausible extrapolation guard
                if np.isfinite(yy).any() and np.nanmax(np.abs(yy)) > 1e8:
                    st.warning("Some displayed values are extremely large. Check units and interval; this may be implausible extrapolation or polynomial growth.")
                # classroom simulation
                st.markdown("#### Contextual visual simulation")
                sim=st.selectbox("Visual simulation",["Walking person with prosthetic leg","Moving object / projectile","Tank or concentration indicator","Population / cell colony","Thermometer / weather","Business / economic indicator"])
                scale=st.slider("Visual scale (display only)",0.5,3.0,1.0,0.1)
                # simple stylized visual tied to normalized f(x), no misleading physical claim
                val=fx if np.isfinite(fx) else 0
                finite=yy[np.isfinite(yy)]
                span=(float(np.nanmax(finite)-np.nanmin(finite)) if len(finite) else 1) or 1
                norm=float(np.clip((val-(float(np.nanmin(finite)) if len(finite) else 0))/span,0,1))
                st.progress(norm,text=f"Model level: {fmt(val)} {yunit} (normalized display only)")
                if sim=="Walking person with prosthetic leg":
                    st.markdown(f"🚶 🦿  **Stride indicator:** {int(20+60*norm)}% of visual scale. This is a symbolic classroom animation, not a biomechanical simulation.")
                elif sim=="Moving object / projectile":
                    st.markdown(f"🎯 **Object position indicator:** {int(100*norm*scale)} display units. The vertical display is normalized; physical motion requires a suitable position-time model.")
                elif sim=="Tank or concentration indicator":
                    st.markdown(f"🧪 **Level indicator:** {int(100*norm)}% of display capacity. This is a visual mapping, not a dosing recommendation.")
                elif sim=="Population / cell colony":
                    st.markdown(f"🦠 **Colony indicator:** {int(10+90*norm)} symbolic cells. Count is scaled for visualization only.")
                elif sim=="Thermometer / weather":
                    st.markdown(f"🌡️ **Indicator:** {int(-10+50*norm)}° display scale; actual units remain {yunit}.")
                else:
                    st.markdown(f"📊 **Economic indicator:** {int(100*norm)}% of the display scale; model output is {fmt(val)} {yunit}.")
                if play:
                    st.info(f"Playback is enabled at {speed} speed. Use the Current input x slider to move the point; browser-side continuous animation is not implemented in this single-file version.")
            except Exception as e:
                st.error(f"Function error: {e}")
                st.caption("Examples: x**2 - 4*x + 3, 2*exp(0.2*x), log(x+2), sin(x), or 1/(x-1). Use * for multiplication and ** or ^ for powers.")

with tab_guide:
    st.subheader("Run locally")
    st.code("pip install -r requirements.txt\npython -m streamlit run app.py",language="bash")
    st.subheader("GitHub and Streamlit Community Cloud")
    st.markdown("1. Create a GitHub repository and upload `app.py` and `requirements.txt` to the repository root.\n2. Commit and push the files.\n3. Open Streamlit Community Cloud, connect GitHub, choose the repository and branch, and set the main file path to `app.py`.\n4. Deploy, then check the app logs if startup fails.")
    st.subheader("Render deployment")
    st.markdown("Create a **Python Web Service**, connect the GitHub repository, and set:")
    st.write("**Build command**")
    st.code("pip install -r requirements.txt")
    st.write("**Start command**")
    st.code("python -m streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT")
    st.subheader("Verification checklist")
    st.markdown("""
- Compare uploaded values with their original source, record the source URL/date, and verify units.
- Edit and reset data; try duplicate years, missing values, nonnumeric cells, and fewer than degree + 1 distinct years.
- Check the displayed equation, centered-time base year, residuals, R², MAE, observation count, range, and mean.
- Confirm single-year and batch estimates match for the same year; test malformed years and years outside 1950–2100.
- Change country, indicator, degree, or data and confirm stale prediction results are invalidated.
- Test each function family, malformed expressions, logarithm/square-root restrictions, and rational discontinuities.
- Move x and the secant endpoints; verify point values, tangent/secant lines, derivative values, and context explanations update together.
- Toggle curves, switch course modes and simulations, and verify exports reflect the current working data and calculated results.
- Test CSV opens as CSV, Excel opens as a workbook with Summary/Working Data/Calculated Years sheets, and PDF opens as a genuine report.
- After deployment, test on a fresh browser session and inspect logs for dependency, port, or parsing errors.
""")
    st.subheader("How the app works")
    st.markdown("""
**Data flow:** built-in demo series or uploaded CSV → editable working table → centered-time polynomial fit → in-sample diagnostics and optional chronological holdout → single/batch estimates → plot and exports.

**Mathematics:** the historical model fits a degree-d polynomial to t = year − base year using least squares. At least d + 1 distinct years are required. R² and MAE are in-sample diagnostics; a chronological holdout is separate. Residuals summarize observed fit errors but are not, by themselves, a validated 95% prediction interval.

**Function lab:** SymPy parses the expression and calculates symbolic derivatives. NumPy evaluates curves; numerical sampling supports graphing and approximate sign-change checks. Exact symbolic derivatives and numerical classifications are explicitly distinguished.

**Limitations:** the bundled historical series are illustrative, not official statistics. Use a sourced dataset and preserve its citation. Polynomial extrapolation can be unstable, numeric root finding can miss complicated roots or discontinuities, and the visual simulations are normalized illustrations rather than physical or medical models. The playback control is a placeholder for slider-driven exploration, not continuous animation. Country comparison demo series are illustrative and may not share real-world coverage.
""")
    st.caption(f"Build generated {datetime.now().strftime('%Y-%m-%d')}. Add source citations and validate domain-specific assumptions before classroom or public use.")
