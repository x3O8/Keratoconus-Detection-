# Workspace interaction contract

## Sources

The user requested a premium frontend to present the existing models. The maintained workflow is `.superdesign/design-system.md`; model and feature contracts are `utils/model_loader.py` and `frontend/src/types.ts`. Evaluation authority is the three JSON reports imported by `frontend/src/data/catalog.ts`. No billing, authentication, retention, or irreversible record lifecycle is implemented by this frontend.

## Canonical UI Map

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| Select/Listbox | PatientData native select | This contract | native | Browser selection and keyboard check |
| Form | PatientData ClinicalField and fieldError | model_loader feature bounds | manual / CSV | Invalid range, example, and upload checks |
| Scrollbar | premium.css global baseline | DESIGN.md and runtime tokens | document | Computed style and mobile overflow check |
| Toast | App notification live region | This contract | export acknowledgement | Browser export check |

Native select ownership is intentional: these are short fixed option lists and operating-system popup geometry and keyboard handling are accepted. The interface uses the same input styling for eye and gender. There are no authored listboxes, date controls, selectable tables, destructive record operations, or modal dialogs.

## Flows and state

- Models is the default landing view. It restores the original headline, one Start assessment action, and an interactive corneal-map illustration with spacious layout. The map is labelled schematic and never represents patient data. Concise model information and saved evaluations remain available below, including while offline. No product name is displayed.
- Reconnect checks status with cancellation and a ten-second timeout. Offline retries automatically every ten seconds; connected status is checked every thirty seconds. Offline is explicit; no fabricated predictions substitute for unavailable models.
- Assessment retains values while navigating to overview and back. The case identifier never goes in the route. Clinical measurements remain in memory; page unload warns when inputs would be lost.
- Manual clinical readiness requires all twelve fields with numeric values within the backend schema bounds. Errors are textual, associated with the field, and exposed through aria-invalid. Missing values are not substituted.
- CSV and image selection accepts non-empty files up to 10 MB, validates supported types, and leaves previous valid input intact after an invalid choice. The server remains the final content validator. Picker, keyboard, and drag/drop entry are supported.
- Clinical templates download locally and work offline. No file uploads occur until Run assessment is selected.
- Run assessment is enabled only for an input pathway whose model is available. Busy state blocks repeat clicks and keeps button geometry. Errors preserve inputs; success selects Results. The selected threshold is preserved without whole-percent rounding.
- Results and Explainability unlock after a real API response. They are labelled with model metadata, independent probabilities, threshold, data quality, and interpretation limits. Back navigation preserves the assessment.
- Export evidence appears only with the evaluation reports. It downloads recorded evaluations, source filenames, and research context as JSON, excluding assessment data. A single polite status message acknowledges completion.
- Evaluation disclosures use buttons with aria-expanded and aria-controls. Native details/summary controls own progressive disclosure for settings, additional clinical contributions, sources, model outputs, input checks, attention, and clinical context. These controls support keyboard activation. Print CSS expands hidden result details. Missing-input warnings and model disagreement stay visible by default.
- TopNav is the single persistent navigation component across desktop and phone layouts. Navigation is the sole persistent entry to assessment, results, and explainability. Only the Models view adds Start assessment. No global toolbar, repeated new-assessment button, or full-width explanation call to action is displayed.
- Hash navigation selects workspace views; anchors such as model-library and main-content do not select a view. Reloading a result route cannot restore sensitive result data and returns to overview.

## Verification

Build: `npm run build` from frontend. Static premium audit is scoped to `frontend/src` via `premium-ui.json`; the legacy `web/` prototype is outside this change. Browser verification covers the offline overview, model evidence disclosures, map controls, export, input preservation, validation, native selection, mobile reflow, and a real clinical assessment with feature explanations and local grounded answers. `tests/test_server.py` verifies actual clinical and imaging model inference, fusion, image explanations, malformed inputs, and production serving. Fixture checks remain separately labelled frontend contract checks.
