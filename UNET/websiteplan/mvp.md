## Plan: Flood Dataset Maker MVP

Build a web MVP that auto-generates flood masks in batch, lets users correct masks (brush/erase/smart-select via SAM), and exports training-ready datasets. Reuse current segmentation checkpoints and notebook inference logic while introducing a clean FastAPI + React architecture designed for async batch jobs and downloadable ZIP exports.

**Steps**
1. Phase 1 - Reuse and harden inference core
2. Extract reusable inference logic from [UNET/UNet4Fold.ipynb](UNET/UNet4Fold.ipynb), [DeepLabV3+.ipynb](DeepLabV3+.ipynb), and [SAM.ipynb](SAM.ipynb) into backend Python modules. Keep one primary auto-mask model (UNet or DeepLab) plus SAM refinement endpoints. This phase blocks all later phases.
3. Define canonical image and mask contract: RGB image input, binary PNG mask output with pixel values 0/255 only, consistent naming, and metadata row generation. This depends on step 2.
4. Phase 2 - Backend API and async job system
5. Implement FastAPI endpoints for upload, job creation, job status polling, per-image mask retrieval, manual edit save, smart-select request, and export ZIP. This depends on step 3.
6. Add background workers for batch inference (100+ images), progress tracking, retries, and failure logging. Use queue-backed jobs so requests do not timeout. This depends on step 5.
7. Add storage layout for originals, generated masks, edited masks, overlays, and metadata manifests. This is parallel with step 6 once endpoint contracts are fixed.
8. Phase 3 - Frontend annotation UX
9. Build React upload page with drag/drop, batch queue, per-image status, and side-by-side viewer (left original, right mask/overlay). Depends on step 5.
10. Add annotation tools: brush, eraser, adjustable brush size, undo/redo, zoom/pan, and save revision states. Depends on step 9.
11. Add smart-selection flow using SAM click prompts and optional box prompts; merge generated region into current mask in editor. Depends on step 10 and backend smart-select endpoint.
12. Phase 4 - Export, QA gates, and packaging
13. Implement download modes: per-image PNG and full ZIP bundle containing images/, masks/, overlays/, metadata.csv, and config.json (threshold/model/version). Depends on steps 6-11.
14. Add dataset quality checks before export: binary-mask validation, missing pair checks, dimensional consistency, and duplicate-name detection. Depends on step 13.
15. Add operational safeguards: file size limits, image count caps, allowed formats, rate limits, and storage cleanup policy for old jobs. Parallel with step 14.
16. Phase 5 - Verification and acceptance
17. Run backend tests for upload/inference/export contracts and mask integrity checks; run frontend interaction tests for editor tools and smart-select latency handling.
18. Execute end-to-end manual test with 120 images: upload, auto-generate, edit 10 masks, export ZIP, then consume exported dataset in training notebook data loader for compatibility validation.

**Relevant files**
- [UNET/UNet4Fold.ipynb](UNET/UNet4Fold.ipynb) - Reuse image/mask I/O patterns, threshold behavior, and metrics utility logic.
- [DeepLabV3+.ipynb](DeepLabV3+.ipynb) - Reuse checkpoint loading and segmentation inference path for robust auto-mask baseline.
- [SAM.ipynb](SAM.ipynb) - Reuse click-based segmentation flow for smart-selection endpoint behavior.
- [YOLO.ipynb](YOLO.ipynb) - Reference thresholding and contour cleanup ideas for postprocessing options.

**Verification**
1. API contract tests: upload -> job create -> status -> mask fetch -> edit save -> export.
2. Mask validity checks: ensure masks contain only 0/255 and pair correctly with originals.
3. Performance target test: 100+ images processed without request timeout using async workers.
4. Frontend tool tests: brush/eraser/smart-select apply correctly and persist after refresh.
5. Export compatibility test: run exported set through notebook preprocessing/data loader without path or shape errors.

**Decisions**
- Included scope: FastAPI + React, batch auto-mask generation, side-by-side editor, brush/eraser, SAM click smart-select, ZIP export.
- Excluded from MVP: multi-user auth/roles, cloud billing, advanced collaboration/review workflow, active-learning retraining loop.
- Preferred first model path: use one stable auto-mask model for consistency; optional model switching can be post-MVP.

**Further Considerations**
1. Main flaw risk: pseudo-label quality drift. Auto-generated masks can look plausible but be wrong, causing noisy training labels.
2. Main UX risk: smart-select expectation mismatch. Users may expect perfect object boundaries; SAM prompts can still leak to similar regions.
3. Main scale risk: browser memory and upload bottlenecks with 100+ high-resolution images; requires chunked upload and server-side resizing policy.

**Flaw/Risk Analysis and Mitigations**
1. Label noise accumulation
Recommendation: enforce review gates (confidence heatmap, changed-pixel summary, and minimum manual spot-check count before export).
2. Tooling complexity in browser canvas
Recommendation: start with single-layer binary mask editing and defer multi-layer/instance editing.
3. Inference latency and queue congestion
Recommendation: async worker queue, progress polling, and optional low-res preview first then full-res finalize.
4. Storage growth and orphan artifacts
Recommendation: TTL cleanup jobs plus explicit archive/export ownership per job.
5. Dataset inconsistency across sessions
Recommendation: persist model version, threshold, and postprocessing settings in export manifest/config.
6. Legal/privacy concerns from user-uploaded flood imagery
Recommendation: clear retention policy and delete-on-demand endpoint in MVP.
