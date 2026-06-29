"""
Skin Toxicity Predictor - Web Version (Render.com deployment)
VEGA-style: Section 1 (Prediction Summary) + 3.1 (Similar Compounds) + 3.2 (AD Scores)
"""

import flet as ft
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import warnings
import datetime
import os
warnings.filterwarnings('ignore')

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw, AllChem, DataStructs
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, Image as RLImage, HRFlowable,
                                 KeepTogether)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

RDLogger.DisableLog('rdApp.*')

# ── Training data ─────────────────────────────────────────────────────────────
_DATA_PATH = Path(__file__).parent / 'final_skin_irritation_sensitization_descriptors.csv'
_TRAIN_DF = None
TASK_MAP = {
    'irritation':  'skin_irritation_qualitative',
    'sens_invivo': 'skin_sensitization_invivo_call',
    'sens_invitro':'skin_sensitization_invitro_call',
}
LABEL_MAP = {0: 'Negative', 1: 'Positive'}
DESC_COLS = None

def _load_train_data():
    global _TRAIN_DF, DESC_COLS
    if _TRAIN_DF is not None:
        return
    try:
        df = pd.read_csv(_DATA_PATH)
        _TRAIN_DF = df
        non_desc = {'task','source_file','source_sheet','Chemical_Name',
                    'SMILES','standardized_smi','label'}
        DESC_COLS = [c for c in df.columns if c not in non_desc]
        print(f"[OK] Training data loaded: {len(df)} rows")
    except Exception as e:
        print(f"[WARNING] Training data not found: {e}")
        _TRAIN_DF = pd.DataFrame()

def _get_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

def compute_ad(query_smiles, task_key, pred_label, desc_dict):
    _load_train_data()
    if _TRAIN_DF is None or len(_TRAIN_DF) == 0:
        return [], {}

    task_name = TASK_MAP.get(task_key, '')
    sub = _TRAIN_DF[_TRAIN_DF['task'] == task_name].copy()
    if len(sub) == 0:
        return [], {}

    q_fp = _get_fp(query_smiles)
    if q_fp is None:
        return [], {}

    sims = []
    for _, row in sub.iterrows():
        fp = _get_fp(str(row['standardized_smi']))
        sims.append(DataStructs.TanimotoSimilarity(q_fp, fp) if fp else 0.0)
    sub = sub.copy()
    sub['_sim'] = sims
    top5 = sub.nlargest(5, '_sim')

    similar = []
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        similar.append({
            'rank': i,
            'smiles': str(row['standardized_smi']),
            'similarity': round(row['_sim'], 3),
            'exp_label': LABEL_MAP.get(int(row['label']), str(row['label'])),
            'chem_name': str(row.get('Chemical_Name', '')),
        })

    max_sim = top5['_sim'].max() if len(top5) > 0 else 0.0
    high_sim = top5[top5['_sim'] >= 0.7]
    sim_index = round(min(max_sim, 1.0), 3)
    acc_index = round((high_sim['label'] == pred_label).sum() / len(high_sim), 3) if len(high_sim) > 0 else 0.5
    conc_index = round((high_sim['label'] == pred_label).sum() / len(high_sim), 3) if len(high_sim) > 0 else 0.0

    if DESC_COLS and len(sub) > 0:
        in_range, total = 0, 0
        for col in DESC_COLS[:50]:
            if col in desc_dict and col in sub.columns:
                val = desc_dict[col]
                mn, mx = sub[col].min(), sub[col].max()
                if pd.notna(mn) and pd.notna(mx):
                    total += 1
                    if mn <= val <= mx:
                        in_range += 1
        desc_range_ok = (in_range / total >= 0.8) if total > 0 else True
    else:
        desc_range_ok = True

    if sim_index >= 0.7 and conc_index >= 0.7 and desc_range_ok:
        global_ad, ad_in = 1.0, True
    elif sim_index >= 0.4 or conc_index >= 0.4:
        global_ad, ad_in = 0.5, False
    else:
        global_ad, ad_in = 0.0, False

    return similar, {
        'global_ad': global_ad, 'ad_in': ad_in,
        'similarity_index': sim_index, 'accuracy_index': acc_index,
        'concordance_index': conc_index, 'desc_range_ok': desc_range_ok,
    }


# ── PDF Generator ─────────────────────────────────────────────────────────────

def generate_pdf(report_data: dict, out_path: str):
    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )
    h1 = ParagraphStyle('h1', fontSize=16, textColor=colors.HexColor('#1F497D'),
                         spaceAfter=10, spaceBefore=6, leading=22, fontName='Helvetica-Bold')
    normal = ParagraphStyle('n', fontSize=10, spaceAfter=6, leading=16, fontName='Helvetica')
    small  = ParagraphStyle('s', fontSize=9, textColor=colors.HexColor('#555555'),
                             spaceAfter=4, leading=14, fontName='Helvetica')

    smiles    = report_data['smiles']
    results   = report_data['results']
    similar   = report_data['similar']
    ad_scores = report_data['ad_scores']
    mol_bytes = report_data['mol_bytes']

    story = []

    story.append(Paragraph("Skin Toxicity Predictor", ParagraphStyle(
        'title', fontSize=22, textColor=colors.HexColor('#1F497D'),
        fontName='Helvetica-Bold', spaceAfter=6, leading=28)))
    story.append(Paragraph("VEGA-style Prediction &amp; Applicability Domain Report",
                             ParagraphStyle('sub', fontSize=11, textColor=colors.HexColor('#666666'),
                                            fontName='Helvetica', spaceAfter=6, leading=16)))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  SMILES: <b>{smiles}</b>",
        small))
    story.append(HRFlowable(width='100%', thickness=1, color=colors.HexColor('#2E74B5'), spaceAfter=12))

    # Section 1
    story.append(Paragraph("1.  Prediction Summary", h1))
    mol_img_flowable = None
    if mol_bytes:
        mol_img_flowable = RLImage(BytesIO(mol_bytes), width=5*cm, height=5*cm)

    def result_cell(title, res):
        label = "Positive" if res['label'] == 1 else "Negative"
        prob  = res['probability'] * 100
        c_hex = '#CC0000' if res['label'] == 1 else '#007700'
        return (f"<b>{title}</b><br/>"
                f"<font color='{c_hex}'><b>{label}</b></font><br/>"
                f"Probability: {prob:.1f}%")

    result_table = Table([
        [mol_img_flowable or Paragraph("(no image)", small),
         Paragraph(result_cell("Skin Irritation", results['irritation']), normal)],
        ['', Paragraph(result_cell("Sensitization (in vivo)", results['sens_invivo']), normal)],
        ['', Paragraph(result_cell("Sensitization (in vitro)", results['sens_invitro']), normal)],
    ], colWidths=[5.5*cm, 11*cm])
    result_table.setStyle(TableStyle([
        ('SPAN', (0,0),(0,2)), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ROWBACKGROUNDS',(1,0),(1,2),[colors.HexColor('#FFF0F0'),
                                       colors.HexColor('#FFF8F0'),
                                       colors.HexColor('#F0F4FF')]),
        ('BOX',(0,0),(-1,-1),0.5,colors.HexColor('#CCCCCC')),
        ('INNERGRID',(0,0),(-1,-1),0.3,colors.HexColor('#DDDDDD')),
        ('TOPPADDING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),12),
        ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"<b>Compound SMILES:</b> {smiles}", normal))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#CCCCCC'), spaceAfter=10))

    # Section 3.1
    story.append(Paragraph("3.1  Applicability Domain: Similar Compounds", h1))
    story.append(Paragraph(
        "Top-5 most similar compounds from training set (Tanimoto / ECFP4, radius=2)", small))
    story.append(Spacer(1, 0.3*cm))

    for comp in similar:
        exp_color = '#CC0000' if comp['exp_label'] == 'Positive' else '#007700'
        sim_mol_img = None
        sim_mol = Chem.MolFromSmiles(comp['smiles'])
        if sim_mol:
            ibuf = BytesIO()
            Draw.MolToImage(sim_mol, size=(160,160)).save(ibuf, format='PNG')
            ibuf.seek(0)
            sim_mol_img = RLImage(ibuf, width=3*cm, height=3*cm)
        info = (f"<b>Compound #{comp['rank']}</b><br/>"
                f"SMILES: {comp['smiles']}<br/>"
                f"Name: {comp['chem_name'] or 'N/A'}<br/>"
                f"Similarity: {comp['similarity']:.3f} ({int(comp['similarity']*100)}%)<br/>"
                f"Experimental value: <font color='{exp_color}'><b>{comp['exp_label']}</b></font>")
        sim_row = Table([[sim_mol_img or Paragraph('',small), Paragraph(info, normal)]],
                        colWidths=[3.5*cm, 13*cm])
        sim_row.setStyle(TableStyle([
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('BOX',(0,0),(-1,-1),0.5,colors.HexColor('#CCCCCC')),
            ('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#FAFAFA')),
            ('TOPPADDING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),12),
            ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),
        ]))
        story.append(KeepTogether(sim_row))
        story.append(Spacer(1, 0.2*cm))

    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#CCCCCC'), spaceAfter=10))

    # Section 3.2
    story.append(Paragraph("3.2  Applicability Domain: AD Scores", h1))
    story.append(Paragraph("Measured Applicability Domain Scores", small))
    story.append(Spacer(1, 0.3*cm))

    def sc(val, good=0.7, bad=0.4):
        if isinstance(val, bool):
            return ('GOOD', colors.HexColor('#007700')) if val else ('BAD', colors.red)
        if val >= good: return ('GOOD', colors.HexColor('#007700'))
        if val >= bad:  return ('CAUTION', colors.HexColor('#CC6600'))
        return ('BAD', colors.red)

    for label, value, explanation, (st_str, st_col) in [
        ("Global AD Index",
         f"AD index = {ad_scores['global_ad']:.1f}",
         "In Domain" if ad_scores['ad_in'] else "Outside Domain",
         sc(ad_scores['global_ad'])),
        ("Similar molecules",
         f"Similarity index = {ad_scores['similarity_index']:.3f}",
         "Strongly similar compounds found." if ad_scores['similarity_index']>=0.7 else "Limited similar compounds.",
         sc(ad_scores['similarity_index'])),
        ("Accuracy for similar molecules",
         f"Accuracy index = {ad_scores['accuracy_index']:.3f}",
         "Prediction accuracy is good." if ad_scores['accuracy_index']>=0.7 else "Prediction accuracy is limited.",
         sc(ad_scores['accuracy_index'])),
        ("Concordance for similar molecules",
         f"Concordance index = {ad_scores['concordance_index']:.3f}",
         "Similar molecules agree with prediction." if ad_scores['concordance_index']>=0.7 else "Limited concordance.",
         sc(ad_scores['concordance_index'])),
        ("Descriptor range check",
         "Range check = " + ("True" if ad_scores['desc_range_ok'] else "False"),
         "Descriptors inside training range." if ad_scores['desc_range_ok'] else "Some descriptors outside range.",
         sc(ad_scores['desc_range_ok'])),
    ]:
        ad_t = Table([[
            Paragraph(f"<b>{label}</b><br/><font size='9'>{value}</font><br/>"
                      f"<font size='9' color='#555555'>{explanation}</font>", normal),
            Paragraph(f"<b>{st_str}</b>", ParagraphStyle('st', fontSize=10,
                      fontName='Helvetica-Bold', textColor=colors.white, alignment=TA_CENTER)),
        ]], colWidths=[13.5*cm, 3*cm])
        ad_t.setStyle(TableStyle([
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('BACKGROUND',(0,0),(0,0),colors.HexColor('#F5F5F5')),
            ('BACKGROUND',(1,0),(1,0),st_col),
            ('BOX',(0,0),(-1,-1),0.5,colors.HexColor('#CCCCCC')),
            ('INNERGRID',(0,0),(-1,-1),0.3,colors.HexColor('#DDDDDD')),
            ('TOPPADDING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),12),
            ('LEFTPADDING',(0,0),(-1,-1),14),('RIGHTPADDING',(0,0),(-1,-1),14),
        ]))
        story.append(KeepTogether(ad_t))
        story.append(Spacer(1, 0.2*cm))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#CCCCCC')))
    story.append(Paragraph(
        "Skin Toxicity Predictor — For research and educational purposes only. "
        "Developed using RDKit + scikit-learn + Flet.",
        ParagraphStyle('footer', fontSize=8, textColor=colors.HexColor('#888888'),
                        fontName='Helvetica', alignment=TA_CENTER, spaceBefore=6)))
    doc.build(story)


# ── Predictor ─────────────────────────────────────────────────────────────────

class SkinToxicityPredictor:
    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent
        self.models = {}
        self.load_models()

    def load_models(self):
        for key, fname in {
            'irritation':  'model_best_skin_irritation_qualitative.joblib',
            'sens_invivo': 'model_best_skin_sensitization_invivo_call.joblib',
            'sens_invitro':'model_best_skin_sensitization_invitro_call.joblib',
        }.items():
            p = self.models_dir / fname
            if p.exists():
                self.models[key] = joblib.load(p)
                print(f"[OK] Loaded {key}")

    def smiles_to_descriptors(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return dict(Descriptors.CalcMolDescriptors(mol))

    def predict(self, smiles):
        if not self.models:
            return None, "No models loaded"
        try:
            desc_dict = self.smiles_to_descriptors(smiles)
            results = {'smiles': smiles, 'descriptors': desc_dict}
            for key, package in self.models.items():
                X = pd.DataFrame([desc_dict]).reindex(
                    columns=package['feature_columns'], fill_value=0)
                results[key] = {
                    'label': int(package['model'].predict(X)[0]),
                    'probability': float(package['model'].predict_proba(X)[0, 1]),
                }
            return results, None
        except Exception as e:
            return None, str(e)

    def generate_mol_image_bytes(self, smiles, size=(250, 250)):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            buf = BytesIO()
            Draw.MolToImage(mol, size=size).save(buf, format='PNG')
            return buf.getvalue()
        except:
            return None


# ── UI helpers ────────────────────────────────────────────────────────────────

def score_icon(val, good_thresh=0.7, bad_thresh=0.4):
    if isinstance(val, bool):
        return ("GOOD", ft.Colors.GREEN) if val else ("BAD", ft.Colors.RED)
    if val >= good_thresh:   return ("GOOD",    ft.Colors.GREEN)
    elif val >= bad_thresh:  return ("CAUTION", ft.Colors.ORANGE)
    else:                    return ("BAD",     ft.Colors.RED)

def make_result_card(title, bgcolor, res_t, prob_t, ad_t):
    return ft.Container(
        content=ft.Column([
            ft.Text(title, size=13, weight=ft.FontWeight.BOLD, color=ft.Colors.GREY_800),
            res_t, prob_t, ad_t,
        ], spacing=4, tight=True),
        bgcolor=bgcolor, padding=12, border_radius=10, expand=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main(page: ft.Page):
    page.title = "Skin Toxicity Predictor"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    page.scroll = "auto"

    predictor = SkinToxicityPredictor()
    _load_train_data()

    smiles_input = ft.TextField(label="SMILES", hint_text="e.g. CCO", width=480)

    def load_example(s):
        def _click(e):
            smiles_input.value = s
            page.update()
        return _click

    example_row = ft.Row([
        ft.TextButton("Ethanol",      on_click=load_example("CCO")),
        ft.TextButton("Acetic acid",  on_click=load_example("CC(=O)O")),
        ft.TextButton("Benzene",      on_click=load_example("c1ccccc1")),
        ft.TextButton("Triethylamine",on_click=load_example("CCN(CC)CC")),
    ], wrap=True)

    status_text = ft.Text("", italic=True, size=12)
    loading = ft.ProgressRing(visible=False, width=24, height=24)
    _report_data = {}

    # PDF export — web 환경에서는 브라우저 다운로드로 처리
    IS_WEB = os.environ.get("PORT") is not None

    def on_export_pdf(e):
        if not _report_data:
            status_text.value = "Please run a prediction first."
            status_text.color = ft.Colors.ORANGE
            page.update()
            return
        try:
            smiles_safe = _report_data['smiles'].replace('/','_').replace('\\','_')[:30]
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            out_path = str(Path(__file__).parent / f"report_{smiles_safe}_{ts}.pdf")
            generate_pdf(_report_data, out_path)
            status_text.value = f"PDF saved: {out_path}"
            status_text.color = ft.Colors.GREEN
            if not IS_WEB:
                os.startfile(out_path)
        except Exception as ex:
            status_text.value = f"PDF error: {ex}"
            status_text.color = ft.Colors.RED
        page.update()

    pdf_btn = ft.ElevatedButton(
        "Export PDF", icon=ft.Icons.PICTURE_AS_PDF,
        on_click=on_export_pdf,
        style=ft.ButtonStyle(bgcolor=ft.Colors.RED_700, color=ft.Colors.WHITE),
        visible=False,
    )

    mol_image = ft.Image(src=b"", width=220, height=220, visible=False)

    def make_trio():
        return (ft.Text("—", size=15, weight=ft.FontWeight.BOLD),
                ft.Text("", size=11, color=ft.Colors.GREY_700),
                ft.Text("", size=11))

    irr_res,  irr_prob,  irr_ad  = make_trio()
    vivo_res, vivo_prob, vivo_ad = make_trio()
    vitro_res,vitro_prob,vitro_ad= make_trio()

    card_irr   = make_result_card("Skin Irritation",         ft.Colors.RED_50,    irr_res,  irr_prob,  irr_ad)
    card_vivo  = make_result_card("Sensitization (in vivo)", ft.Colors.ORANGE_50, vivo_res, vivo_prob, vivo_ad)
    card_vitro = make_result_card("Sensitization (in vitro)",ft.Colors.BLUE_50,   vitro_res,vitro_prob,vitro_ad)

    similar_column  = ft.Column([], spacing=8)
    ad_score_column = ft.Column([], spacing=6)

    def _ad_score_row(label, value_str, status, color):
        return ft.Container(
            content=ft.Row([
                ft.Container(
                    content=ft.Text(status, size=11, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                    bgcolor=color, border_radius=4, padding=ft.Padding(8,4,8,4), width=70,
                ),
                ft.Column([
                    ft.Text(label, size=12, weight=ft.FontWeight.BOLD),
                    ft.Text(value_str, size=11, color=ft.Colors.GREY_700),
                ], spacing=1, tight=True),
            ], spacing=10),
            bgcolor=ft.Colors.GREY_50, border_radius=8, padding=10,
        )

    def on_predict(e):
        smiles = smiles_input.value.strip()
        if not smiles:
            status_text.value = "Please enter a SMILES code"
            status_text.color = ft.Colors.RED
            page.update()
            return

        loading.visible = True
        status_text.value = "Predicting..."
        status_text.color = ft.Colors.BLUE
        similar_column.controls.clear()
        ad_score_column.controls.clear()
        page.update()

        img_bytes = predictor.generate_mol_image_bytes(smiles)
        if img_bytes:
            mol_image.src = img_bytes
            mol_image.visible = True

        results, error = predictor.predict(smiles)
        loading.visible = False

        if error:
            status_text.value = f"Error: {error}"
            status_text.color = ft.Colors.RED
            page.update()
            return

        def fill_card(res_t, prob_t, ad_t, res, task_key):
            lbl = res['label']
            res_t.value = "Positive" if lbl == 1 else "Negative"
            res_t.color = ft.Colors.RED if lbl == 1 else ft.Colors.GREEN
            prob_t.value = f"Probability: {res['probability']*100:.1f}%"
            _, ads = compute_ad(smiles, task_key, lbl, results.get('descriptors', {}))
            if ads:
                ad_t.value  = "AD: In Domain" if ads['ad_in'] else "AD: Outside Domain"
                ad_t.color  = ft.Colors.GREEN if ads['ad_in'] else ft.Colors.ORANGE

        fill_card(irr_res,  irr_prob,  irr_ad,  results['irritation'],  'irritation')
        fill_card(vivo_res, vivo_prob, vivo_ad, results['sens_invivo'], 'sens_invivo')
        fill_card(vitro_res,vitro_prob,vitro_ad,results['sens_invitro'],'sens_invitro')

        similar, ad_scores = compute_ad(
            smiles, 'irritation', results['irritation']['label'],
            results.get('descriptors', {}))

        for comp in similar:
            sim_bytes = predictor.generate_mol_image_bytes(comp['smiles'], size=(120,120))
            img_ctrl = ft.Image(src=sim_bytes, width=110, height=110) if sim_bytes else ft.Container(width=110, height=110)
            exp_color = ft.Colors.RED if comp['exp_label'] == 'Positive' else ft.Colors.GREEN
            similar_column.controls.append(ft.Container(
                content=ft.Row([
                    img_ctrl,
                    ft.Column([
                        ft.Text(f"Compound #{comp['rank']}", weight=ft.FontWeight.BOLD, size=12),
                        ft.Text(f"SMILES: {comp['smiles']}", size=11, color=ft.Colors.GREY_700,
                                max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                        ft.Text(comp['chem_name'], size=11, color=ft.Colors.GREY_600,
                                max_lines=1, overflow=ft.TextOverflow.ELLIPSIS),
                        ft.Row([
                            ft.Text("Similarity:", size=11),
                            ft.ProgressBar(value=comp['similarity'], width=120,
                                           color=ft.Colors.BLUE_400, bgcolor=ft.Colors.BLUE_50),
                            ft.Text(f"{comp['similarity']:.3f}", size=11, weight=ft.FontWeight.BOLD),
                        ], spacing=6),
                        ft.Row([
                            ft.Text("Exp. value:", size=11),
                            ft.Text(comp['exp_label'], size=11, weight=ft.FontWeight.BOLD, color=exp_color),
                        ], spacing=6),
                    ], spacing=3, expand=True),
                ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.CENTER),
                bgcolor=ft.Colors.WHITE,
                border=ft.Border(ft.BorderSide(1, ft.Colors.GREY_200),
                                 ft.BorderSide(1, ft.Colors.GREY_200),
                                 ft.BorderSide(1, ft.Colors.GREY_200),
                                 ft.BorderSide(1, ft.Colors.GREY_200)),
                border_radius=8, padding=10,
            ))

        if ad_scores:
            for lbl, desc, st, col in [
                ("Global AD Index",
                 f"AD index = {ad_scores['global_ad']:.1f}  |  " +
                 ("In Applicability Domain" if ad_scores['ad_in'] else "OUTSIDE Applicability Domain"),
                 *score_icon(ad_scores['global_ad'])),
                ("Similar molecules with known experimental value",
                 f"Similarity index = {ad_scores['similarity_index']:.3f}  |  " +
                 ("Strongly similar compounds found." if ad_scores['similarity_index']>=0.7 else "Limited similar compounds."),
                 *score_icon(ad_scores['similarity_index'])),
                ("Accuracy of prediction for similar molecules",
                 f"Accuracy index = {ad_scores['accuracy_index']:.3f}  |  " +
                 ("Accuracy is good." if ad_scores['accuracy_index']>=0.7 else "Accuracy is limited."),
                 *score_icon(ad_scores['accuracy_index'])),
                ("Concordance for similar molecules",
                 f"Concordance index = {ad_scores['concordance_index']:.3f}  |  " +
                 ("Similar molecules agree with prediction." if ad_scores['concordance_index']>=0.7 else "Limited concordance."),
                 *score_icon(ad_scores['concordance_index'])),
                ("Descriptor range check",
                 "Range check = " + ("True  |  Inside training set range." if ad_scores['desc_range_ok']
                                     else "False  |  Some descriptors outside range."),
                 *score_icon(ad_scores['desc_range_ok'])),
            ]:
                ad_score_column.controls.append(_ad_score_row(lbl, desc, st, col))

        _report_data.clear()
        _report_data.update({
            'smiles': smiles, 'results': results,
            'similar': similar, 'ad_scores': ad_scores, 'mol_bytes': img_bytes,
        })
        pdf_btn.visible = True
        status_text.value = "Prediction complete!"
        status_text.color = ft.Colors.GREEN
        page.update()

    predict_btn = ft.ElevatedButton(
        "Predict", icon=ft.Icons.SCIENCE, on_click=on_predict,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE_700, color=ft.Colors.WHITE),
    )

    page.add(ft.ListView(expand=True, spacing=0, padding=0, controls=[
        ft.Container(
            content=ft.Column([
                ft.Text("Skin Toxicity Predictor", size=28, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_900),
                ft.Text("VEGA-style Prediction & Applicability Domain Report", size=13, color=ft.Colors.GREY_600),
            ], spacing=4),
            padding=20, bgcolor=ft.Colors.BLUE_50,
        ),
        ft.Divider(height=1),
        ft.Container(
            content=ft.Column([
                ft.Text("Input", size=18, weight=ft.FontWeight.BOLD),
                smiles_input,
                ft.Text("Examples:", size=11, color=ft.Colors.GREY_600),
                example_row,
                ft.Row([predict_btn, loading, pdf_btn], spacing=12),
                status_text,
            ], spacing=8),
            padding=20,
        ),
        ft.Divider(height=1),
        ft.Container(
            content=ft.Column([
                ft.Text("1.  Prediction Summary", size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_800),
                ft.Row([
                    ft.Container(content=mol_image, alignment=ft.Alignment(0,0),
                                 width=240, height=240,
                                 border=ft.Border(ft.BorderSide(1,ft.Colors.GREY_200),
                                                  ft.BorderSide(1,ft.Colors.GREY_200),
                                                  ft.BorderSide(1,ft.Colors.GREY_200),
                                                  ft.BorderSide(1,ft.Colors.GREY_200)),
                                 border_radius=8, bgcolor=ft.Colors.WHITE),
                    ft.Column([card_irr, card_vivo, card_vitro], spacing=8, expand=True),
                ], spacing=16, vertical_alignment=ft.CrossAxisAlignment.START),
            ], spacing=12),
            padding=20,
        ),
        ft.Divider(height=1),
        ft.Container(
            content=ft.Column([
                ft.Text("3.1  Applicability Domain: Similar Compounds",
                        size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_800),
                ft.Text("Top-5 most similar compounds from training set (Tanimoto / ECFP4)",
                        size=12, color=ft.Colors.GREY_600),
                similar_column,
            ], spacing=10),
            padding=20,
        ),
        ft.Divider(height=1),
        ft.Container(
            content=ft.Column([
                ft.Text("3.2  Applicability Domain: AD Scores",
                        size=18, weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_800),
                ft.Text("Measured Applicability Domain Scores", size=12, color=ft.Colors.GREY_600),
                ad_score_column,
            ], spacing=10),
            padding=20,
        ),
    ]))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, port=port, host="0.0.0.0")
