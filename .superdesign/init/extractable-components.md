# Extractable Components

There are no existing React/Vue/Svelte components or reusable frontend layout components to convert into Superdesign DraftComponents. The current browser shell is static HTML and should be used as context, not extracted as a component. Component extraction is therefore skipped for this new React target.

Future React implementation candidates:

## AppSidebar
- Source: new target
- Category: layout
- Description: Product navigation, analysis progress, model-health summary, research-use label
- Extractable props: currentTab, imageModelReady, clinicalModelReady
- Hardcoded: KeratoScan branding, navigation labels, icons

## AnalysisStepper
- Source: new target
- Category: basic
- Description: Patient Data → Results → Explainability progress and navigation
- Extractable props: currentStep, completedSteps
- Hardcoded: step labels and order
