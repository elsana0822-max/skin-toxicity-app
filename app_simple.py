"""
Skin Toxicity Predictor - Simple Stable Version
안정적이고 간단한 버전
"""

import flet as ft
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import tempfile

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Draw

RDLogger.DisableLog('rdApp.*')


class SkinToxicityPredictor:
    """피부 독성 예측 모델 관리"""

    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else Path.cwd()
        self.models = {}
        self.load_models()

    def load_models(self):
        """모델 3개 로드"""
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

    def smiles_to_descriptors(self, smiles):
        """SMILES → Descriptors"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return dict(Descriptors.CalcMolDescriptors(mol))

    def predict(self, smiles):
        """단일 SMILES 예측"""
        if not self.models:
            return None, "No models loaded"

        try:
            desc_dict = self.smiles_to_descriptors(smiles)
            results = {'smiles': smiles, 'descriptors': desc_dict}

            # 주요 descriptor
            results['mol_weight'] = desc_dict.get('MolWt', 0)
            results['log_p'] = desc_dict.get('MolLogP', 0)
            results['tpsa'] = desc_dict.get('TPSA', 0)

            for key, package in self.models.items():
                model = package['model']
                feature_cols = package['feature_columns']

                X = pd.DataFrame([desc_dict]).reindex(columns=feature_cols, fill_value=0)
                pred_label = int(model.predict(X)[0])
                pred_proba = float(model.predict_proba(X)[0, 1])

                results[key] = {
                    'label': pred_label,
                    'probability': pred_proba,
                }

            return results, None

        except Exception as e:
            return None, str(e)

    def generate_mol_image(self, smiles, size=(300, 300)):
        """분자 이미지 생성"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            img = Draw.MolToImage(mol, size=size)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            img.save(temp_file.name)
            temp_file.close()
            return temp_file.name

        except Exception as e:
            print(f"Image error: {e}")
            return None


def main(page: ft.Page):
    """메인 애플리케이션"""

    page.title = "Skin Toxicity Predictor"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 20
    page.window_width = 1000
    page.window_height = 800
    page.scroll = "auto"

    predictor = SkinToxicityPredictor()

    # UI Components
    smiles_input = ft.TextField(
        label="SMILES Code",
        hint_text="예: CCO",
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
    ])

    mol_image = ft.Image(src="", width=300, height=300)

    # Result text controls
    result_irritation = ft.Text("—", size=16, weight=ft.FontWeight.BOLD)
    result_sens_vivo = ft.Text("—", size=16, weight=ft.FontWeight.BOLD)
    result_sens_vitro = ft.Text("—", size=16, weight=ft.FontWeight.BOLD)

    status_text = ft.Text("", italic=True)
    loading_indicator = ft.ProgressRing(visible=False, width=30, height=30)

    def on_predict(e):
        smiles = smiles_input.value.strip()

        if not smiles:
            status_text.value = "Please enter SMILES"
            status_text.color = ft.Colors.RED
            page.update()
            return

        loading_indicator.visible = True
        status_text.value = "Predicting..."
        status_text.color = ft.Colors.BLUE
        page.update()

        # Generate image
        img_path = predictor.generate_mol_image(smiles)
        if img_path:
            mol_image.src = img_path

        # Predict
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

        # Update results
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

    predict_btn = ft.TextButton(
        "🔬 Predict",
        on_click=on_predict,
    )

    # Layout with ListView for scrolling
    page.add(
        ft.ListView(
            controls=[
                ft.Container(
                    content=ft.Column([
                        ft.Text(
                            "🧪 Skin Toxicity Predictor",
                            size=32,
                            weight=ft.FontWeight.BOLD,
                            color=ft.Colors.BLUE_900,
                        ),
                        ft.Text(
                            "ML-based prediction for skin irritation and sensitization",
                            size=14,
                            color=ft.Colors.GREY_700,
                        ),
                    ], spacing=5),
                    padding=20,
                ),

                ft.Divider(),

                ft.Container(
                    content=ft.Column([
                        ft.Text("Input", size=20, weight=ft.FontWeight.BOLD),
                        smiles_input,
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
                                bgcolor=ft.Colors.RED_50,
                                padding=15,
                                border_radius=10,
                            ),
                            ft.Container(
                                content=ft.Column([
                                    ft.Text("Sensitization (in vivo)", size=14, color=ft.Colors.GREY_700),
                                    result_sens_vivo,
                                ]),
                                bgcolor=ft.Colors.ORANGE_50,
                                padding=15,
                                border_radius=10,
                            ),
                            ft.Container(
                                content=ft.Column([
                                    ft.Text("Sensitization (in vitro)", size=14, color=ft.Colors.GREY_700),
                                    result_sens_vitro,
                                ]),
                                bgcolor=ft.Colors.BLUE_50,
                                padding=15,
                                border_radius=10,
                            ),
                        ], spacing=10),
                    ], spacing=15),
                    padding=20,
                ),
            ],
            expand=True,
            spacing=0,
            padding=0,
        )
    )


if __name__ == "__main__":
    ft.app(target=main)
