"""
Skin Toxicity Predictor - Web Version (Render.com deployment)
"""

import flet as ft
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import base64
from io import BytesIO
import os

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw

RDLogger.DisableLog('rdApp.*')


class SkinToxicityPredictor:

    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent
        self.models = {}
        self.load_models()

    def load_models(self):
        model_files = {
            'irritation': 'model_best_skin_irritation_qualitative.joblib',
            'sens_invivo': 'model_best_skin_sensitization_invivo_call.joblib',
            'sens_invitro': 'model_best_skin_sensitization_invitro_call.joblib',
        }
        for key, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                self.models[key] = joblib.load(model_path)
                print(f"[OK] Loaded {key}")
            else:
                print(f"[WARNING] Model not found: {model_path}")

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
            results = {'smiles': smiles}
            for key, package in self.models.items():
                model = package['model']
                feature_cols = package['feature_columns']
                X = pd.DataFrame([desc_dict]).reindex(columns=feature_cols, fill_value=0)
                pred_label = int(model.predict(X)[0])
                pred_proba = float(model.predict_proba(X)[0, 1])
                results[key] = {'label': pred_label, 'probability': pred_proba}
            return results, None
        except Exception as e:
            return None, str(e)

    def generate_mol_image_b64(self, smiles, size=(300, 300)):
        """분자 이미지 → base64 (웹 호환)"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            img = Draw.MolToImage(mol, size=size)
            buf = BytesIO()
            img.save(buf, format='PNG')
            return base64.b64encode(buf.getvalue()).decode()
        except Exception as e:
            print(f"Image error: {e}")
            return None


def main(page: ft.Page):
    page.title = "Skin Toxicity Predictor"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    page.scroll = "auto"

    predictor = SkinToxicityPredictor()

    smiles_input = ft.TextField(
        label="SMILES Code",
        hint_text="e.g. CCO",
        width=500,
    )

    def load_example(smiles):
        def on_click(e):
            smiles_input.value = smiles
            page.update()
        return on_click

    example_row = ft.Row([
        ft.TextButton("Ethanol (CCO)", on_click=load_example("CCO")),
        ft.TextButton("Acetic acid", on_click=load_example("CC(=O)O")),
        ft.TextButton("Benzene", on_click=load_example("c1ccccc1")),
        ft.TextButton("Triethylamine", on_click=load_example("CCN(CC)CC")),
    ], wrap=True)

    # base64 이미지 사용 (웹 호환)
    mol_image = ft.Image(src_base64="", width=300, height=300, visible=False)

    result_irritation = ft.Text("—", size=16, weight=ft.FontWeight.BOLD)
    result_sens_vivo = ft.Text("—", size=16, weight=ft.FontWeight.BOLD)
    result_sens_vitro = ft.Text("—", size=16, weight=ft.FontWeight.BOLD)

    status_text = ft.Text("", italic=True)
    loading_indicator = ft.ProgressRing(visible=False, width=30, height=30)

    def on_predict(e):
        smiles = smiles_input.value.strip()
        if not smiles:
            status_text.value = "Please enter a SMILES code"
            status_text.color = ft.Colors.RED
            page.update()
            return

        loading_indicator.visible = True
        status_text.value = "Predicting..."
        status_text.color = ft.Colors.BLUE
        page.update()

        b64 = predictor.generate_mol_image_b64(smiles)
        if b64:
            mol_image.src_base64 = b64
            mol_image.visible = True

        results, error = predictor.predict(smiles)
        loading_indicator.visible = False

        if error:
            status_text.value = f"Error: {error}"
            status_text.color = ft.Colors.RED
            result_irritation.value = "—"
            result_sens_vivo.value = "—"
            result_sens_vitro.value = "—"
            page.update()
            return

        def format_result(res):
            label = "Positive" if res['label'] == 1 else "Negative"
            prob = res['probability'] * 100
            icon = "⚠" if res['label'] == 1 else "✓"
            return f"{icon} {label} ({prob:.1f}%)"

        result_irritation.value = format_result(results['irritation'])
        result_irritation.color = ft.Colors.RED if results['irritation']['label'] == 1 else ft.Colors.GREEN

        result_sens_vivo.value = format_result(results['sens_invivo'])
        result_sens_vivo.color = ft.Colors.RED if results['sens_invivo']['label'] == 1 else ft.Colors.GREEN

        result_sens_vitro.value = format_result(results['sens_invitro'])
        result_sens_vitro.color = ft.Colors.RED if results['sens_invitro']['label'] == 1 else ft.Colors.GREEN

        status_text.value = "Prediction complete!"
        status_text.color = ft.Colors.GREEN
        page.update()

    predict_btn = ft.TextButton("🔬 Predict", on_click=on_predict)

    page.add(
        ft.ListView(
            controls=[
                ft.Container(
                    content=ft.Column([
                        ft.Text("🧪 Skin Toxicity Predictor", size=32,
                                weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_900),
                        ft.Text("ML-based prediction for skin irritation and sensitization",
                                size=14, color=ft.Colors.GREY_700),
                    ], spacing=5),
                    padding=20,
                ),
                ft.Divider(),
                ft.Container(
                    content=ft.Column([
                        ft.Text("Input", size=20, weight=ft.FontWeight.BOLD),
                        smiles_input,
                        ft.Text("Examples:", size=12, color=ft.Colors.GREY_600),
                        example_row,
                        ft.Row([predict_btn, loading_indicator]),
                        status_text,
                    ], spacing=10),
                    padding=20,
                ),
                ft.Divider(),
                ft.Container(
                    content=ft.Column([
                        ft.Text("Results", size=20, weight=ft.FontWeight.BOLD),
                        ft.Column([
                            ft.Container(
                                content=ft.Column([
                                    ft.Text("Molecule Structure", weight=ft.FontWeight.BOLD),
                                    mol_image,
                                ]),
                                alignment=ft.Alignment(0, 0),
                            ),
                            ft.Container(height=20),
                            ft.Text("Predictions", weight=ft.FontWeight.BOLD),
                            ft.Container(
                                content=ft.Column([
                                    ft.Text("Skin Irritation", size=14, color=ft.Colors.GREY_700),
                                    result_irritation,
                                ]),
                                bgcolor=ft.Colors.RED_50, padding=15, border_radius=10,
                            ),
                            ft.Container(
                                content=ft.Column([
                                    ft.Text("Sensitization (in vivo)", size=14, color=ft.Colors.GREY_700),
                                    result_sens_vivo,
                                ]),
                                bgcolor=ft.Colors.ORANGE_50, padding=15, border_radius=10,
                            ),
                            ft.Container(
                                content=ft.Column([
                                    ft.Text("Sensitization (in vitro)", size=14, color=ft.Colors.GREY_700),
                                    result_sens_vitro,
                                ]),
                                bgcolor=ft.Colors.BLUE_50, padding=15, border_radius=10,
                            ),
                        ], spacing=10),
                    ], spacing=15),
                    padding=20,
                ),
                ft.Container(
                    content=ft.Text(
                        "Developed using Flet + RDKit + scikit-learn",
                        size=10, color=ft.Colors.GREY_500,
                    ),
                    padding=ft.padding.only(left=20, bottom=20),
                ),
            ],
            expand=True, spacing=0, padding=0,
        )
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    ft.app(target=main, view=ft.AppView.WEB_BROWSER, port=port, host="0.0.0.0")
