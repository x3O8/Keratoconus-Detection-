# Frontend verification — 2026-10-02

## Static checks

- TypeScript check and Vite production build pass via `npm run build`.
- Strict premium audit passes, scoped to frontend/src using the root premium-ui.json; evidence is saved in root premium-audit.json.
- DESIGN.md validator passes with zero errors.

## Actual browser checks

- Offline overview renders all model evidence without a service or fabricated predictions.
- Curvature/Thickness map selection changes both contour geometry and accessible description.
- EfficientNetV2-S disclosure exposes external-test scope, checkpoint distinction, threshold, and zero specificity.
- Clinical example fills the twelve fields. Age 150 shows the associated range error and aria-invalid; correcting it restores valid input state.
- Eye selector accepts OS and OD. Labels are associated with native controls.
- Inputs persist across overview/assessment navigation. A valid clinical form remains unable to run while its model is offline.
- Unsupported .txt selection in the CSV upload shows an app-owned correction error and retains the existing form.
- Phone viewport reflows into stacked cards and horizontal navigation; inspected document width does not exceed the viewport. Global scrollbar styling is present in computed styles.

## Isolated API fixture checks

The test-only Vite server at port 5174 proxies fixture_api.py at port 18000. The normal preview at 5173 continues to point at the real service at 8000. Fixtures label every output as synthetic and never run a model.

- Clinical readiness enables Run assessment only after complete valid input.
- Pending submission shows the fixed-width busy button and disables repeat submission.
- First request returns HTTP 422: the readable error appears and entered data remains intact.
- Retry succeeds: Results unlocks and renders clinical-only finding, model metadata, unavailable imaging, single-evidence-stream state, and data-quality information.
- Explanation navigation renders the feature contribution, interpretation limits, and unavailable-image state.
- Explanation assistant displays a completed synthetic response; no frontend console errors were observed in that flow.

## Real inference and combined website

- Restored `server.py` and the original imaging checkpoint under `models/model1.pth`. `python run.py` serves the compiled interface and real inference API together at port 8000; port 5173 proxies that service during development.
- Production `/api/status` reports both CatBoost and EfficientNet-B0 ready, CPU inference, and the stored 0.4293 clinical threshold. The open development browser connected automatically when the service started.
- The browser loaded the built-in twelve-field example and ran the actual clinical model. Results unlocked, showed the selected threshold, clinical-only assessment, unavailable imaging, and data-quality flags. This was repeated on the combined production website with case label `Example verification`.
- Explainability displayed nonzero real TreeSHAP contributions: thinnest pachymetry, Kmax, and anterior asphericity were the leading positive contributions for this example.
- Ask evidence returned an assessment-grounded local answer and linked sources without a Gemini key. Optional external generation is separately disclosed in the interface.
- `python -X utf8 -m pytest tests/test_server.py -q -p no:cacheprovider`: **14 passed**. Tests compare HTTP clinical predictions with direct saved-pipeline predictions; exercise CSV parity and malformed inputs; run the actual imaging checkpoint on synthetic pixels; verify weighted fusion, Grad-CAM PNG overlays, integrated-gradient PNG overlays, and normalized attention; check explanations, the CSV template, and compiled page serving.
- Final TypeScript/Vite build and strict premium audit pass with zero audit findings.

## Unbranded layout simplification

- Removed product-name text, branded browser titles and download filenames, sidebar marketing content, promotional hero, facts strip, repeated page introductions, global toolbar, duplicate assessment buttons, and explanation call-to-action card.
- Models has one Start assessment action. Export is alongside evaluations; Print is alongside results. The four workspace destinations remain available in the rail or sticky mobile navigation.
- Clinical-only results show one probability summary. Full model outputs and input checks use native disclosures. Missing-map warnings and model disagreement remain immediately visible.
- Explainability shows the six strongest TreeSHAP contributors. Keyboard Enter expands the remaining six; sources, attention, and clinical context are available on demand. The question form returns a real local grounded answer.
- Browser verified the simplified form with age 150: range error appeared and Run assessment was disabled; correcting to 27 allowed a real assessment. Analysis settings preserved the stored 42.93% threshold and keyboard disclosure behavior.
- Narrow viewport check: measured document and viewport widths both 422px, with no horizontal overflow and no branding in rendered text. Restored the normal viewport afterward.
- Final build/typecheck, strict premium audit, and official DESIGN.md lint pass. The real-model integration suite remains **14 passed**.

## Restored spacious landing hero

- Restored the original “Small details. Deeper insight.” headline and interactive corneal-map illustration, with one Start assessment button. Product branding, repeated introductions, and duplicate assessment actions remain absent.
- The landing variant adds 52px desktop hero insets, a 500px minimum hero height, and 56px gaps before the model cards and evaluations. Phone layout stacks the headline/action and illustration with generous vertical separation.
- Browser checked Thickness selection changes the accessible map description; Start assessment opens the working form and preserves existing inputs. Returning to Models restores the landing view.
- Phone check measured viewport and document at 422px each, a single hero column, and exactly one Start assessment button. The normal viewport was restored afterward.
- TypeScript/Vite build and strict premium audit pass.

## Top navigation and landing metrics

- Replaced the desktop side rail and separate phone navigation with one sticky TopNav component. All four destinations retain their active state and result-availability rules. The main workspace has zero left offset.
- Clinical cards display repository-backed locked internal test AUROC 0.9983 and AUPRC 0.9972. Imaging cards display saved checkpoint validation accuracy 100.0% and F1 1.0000 from live API metadata; missing metadata renders a dash. The separate EfficientNetV2-S external report remains separately labelled.
- Browser verified one navigation at desktop and phone widths, Start assessment opening the form, keyboard Enter activating Assessment, and the built-in example age 27 retained across Models/Assessment navigation.
- Phone viewport and document widths both measured 422px, with four visible navigation labels and no horizontal overflow. Temporary viewport override was reset afterward.
- TypeScript/Vite production build and strict premium audit pass with zero findings.

## Slim navigation refinement

- Replaced the heavy emerald band, navigation icons, filled active tile, and repeated model-health indicators with a white text-only navigation bar and two-pixel emerald active underline. Desktop bar is 56px; phone bar is 52px. The connection strip is 32px.
- Model readiness and recorded metrics remain on the model cards. The shared navigation preserves result gating, route updates, active-page semantics, and keyboard access.
- Browser verified the white bar and two-pixel active underline, keyboard Enter opening Assessment with a visible focus outline, and the existing example measurements retained. At the narrow breakpoint, document and viewport were both 322px and every navigation label fit. The normal viewport was restored.
- TypeScript/Vite production build, strict premium audit, and official DESIGN.md lint pass with no errors or warnings.

## Clinical validation limits

This verifies software behavior, not clinical performance or external model validity. Imaging tests use synthetic pixels, not patient material. The clinical pipeline emits a scikit-learn 1.6.1 serialization versus 1.9.0 runtime warning; API/direct-pipeline parity does not establish training-environment equivalence. Automated screen-reader/contrast audits and real device touch checks were not run. Initial fixture checks above are retained as historical frontend-contract evidence and are separate from these real-model checks.
