# KeratoScan AI Design System

## Product context

KeratoScan AI is a research clinical decision-support dashboard for ophthalmologists and corneal researchers. It evaluates keratoconus using two independent paths: a multi-modal corneal image model (`model1.pth`) and an XGBoost clinical model (`xgb_final.pkl`, 12 numeric inputs), then presents individual and fused probabilities. It must clearly separate algorithmic assessment from diagnosis.

## Jobs to be done

1. Start a single-patient assessment without needing technical knowledge of the model.
2. Add whatever data is available: patient demographics/clinical topography values manually or by CSV, plus one or more supported corneal maps.
3. See input completeness and validation before inference.
4. Compare image-model, clinical-model, and combined outputs without implying false certainty.
5. Understand the patient-specific drivers through SHAP-style feature contributions and Grad-CAM/modality attention.
6. Export or print a concise research report while keeping the research-only limitation visible.

## Information architecture

- Persistent left sidebar: brand, tabs, current case completeness, model health, research-use label.
- Tab 1 — Patient Data: patient identifier (optional/non-identifying), laterality, manual clinical form, CSV alternative, seven image slots, input validation, model-selection/fusion control, run analysis.
- Tab 2 — Results: decision-support finding, risk probability with uncertainty language, two model cards, fused assessment, disagreement warning, threshold context, recommended clinician review.
- Tab 3 — Explainability: plain-language summary, clinical feature contribution waterfall, top drivers with patient values and direction, image modality attention, Grad-CAM gallery, model limitations.

## Visual language

Professional, calm, and evidence-oriented. Avoid futuristic neon, glossy AI clichés, oversized gradients, or alarmist red. The interface should resemble a modern diagnostic workstation with generous whitespace and clear hierarchy.

### Colors

- App background `#F4F7F6`
- Primary surface `#FFFFFF`
- Primary ink `#102521`
- Muted ink `#6D7D79`
- Hairline `#DFE7E4`
- Brand/primary `#0B6B57`
- Dark navigation `#0F2824`
- Positive/mint surface `#D9F0E9`
- Teal visualization `#2A9D8F`
- Amber/borderline `#C47A2C`, surface `#FFF7E8`
- High-risk red `#B23A3A`, surface `#FFF0F0`; use only for findings and errors
- Informational blue `#376C8A`, surface `#EEF6FA`

### Typography

- Headings: Manrope, 600–700
- Body and controls: DM Sans, 400–600
- Numeric probabilities and clinical values: DM Sans tabular numerals, 600–700
- Page title 28px; section title 20px; card title 15px; body 14px; caption/eyebrow 11px.

### Shape and elevation

- 4px spacing grid.
- Cards: 16px radius, 1px border, no shadow by default.
- Inputs/drop zones: 12px radius.
- Buttons: 10–12px radius, minimum 44px height.
- Pills: full radius; compact and label-first.
- One restrained shadow is allowed for the main result summary and dialogs.

## Core component styles

- Sidebar: 248px, dark green, white primary labels, desaturated mint secondary labels. Active tab uses a lighter translucent green block and a 3px mint indicator.
- Step tabs: numbered icons plus label; show complete, active, and locked/future states without hiding navigation.
- Upload tiles: dashed border while empty; image thumbnail, filename, modality label, and replace/remove actions when populated.
- Clinical form: grouped into Demographics, Keratometry/Astigmatism, Pachymetry, and Asphericity. Each field shows unit, acceptable format, and validation state.
- Probability cards: show model name, KC probability, predicted class, and data source. Avoid displaying confidence as synonymous with clinical certainty.
- Fused finding: large probability plus explicit threshold and text such as “screen positive” or “screen negative”; always include “requires ophthalmologist review.”
- Explainability bars: diverging contributions, teal pushing toward KC and slate pushing toward normal, with patient value and units alongside each label.
- Grad-CAM gallery: original/overlay comparison, selected modality tabs, intensity legend, and a note that highlighted regions are not lesions.

## Motion

- 150–220ms ease-out for tab transitions, upload-state changes, and progress fills.
- Respect `prefers-reduced-motion`.
- No looping or decorative animation.

## Accessibility and safety

- WCAG AA contrast; never rely on color alone.
- Keyboard-visible focus rings in `#2A9D8F`.
- Inline validation associated with fields.
- All uploaded-image previews have modality-specific alt text.
- Research-only banner remains visible on Results and Explainability.
- No diagnosis/treatment recommendation language; use “model assessment,” “screen positive/negative,” and “review.”

## Responsive behavior

- Desktop-first clinical workstation at 1440×1024.
- At <1024px sidebar collapses to a top navigation and two-column sections become one column.
- At <640px cards and upload grid stack; primary action stays accessible without covering content.

## Non-negotiable functional requirements

- Support seven image modalities: CT_A, EC_A, EC_P, Elv_A, Elv_P, Sag_A, Sag_P.
- Support manual entry and CSV import for the 12 clinical features: age_years, gender, astig_value_d, astig_axis_deg, kmax_value_d, kmax_axis_deg, pachy_central_um, pachy_thinnest_um, pachy_thinnest_x, pachy_thinnest_y, asphericity_anterior, asphericity_posterior.
- Clearly show which model(s) can run from currently available data.
- Never silently substitute missing clinical values.
- Treat missing image modalities as missing and disclose any model-side zero filling.
- Explanations must be patient-specific; global feature importance can be secondary context only.
