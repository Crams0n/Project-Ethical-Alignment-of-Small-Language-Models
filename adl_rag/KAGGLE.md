# Running on Kaggle (free GPU)

The code is device-agnostic: `src/llm.py` automatically uses a GPU when one is
available, otherwise CPU. On Kaggle's free **T4** the full run drops from
~4–8 h (CPU) to roughly **15–30 min**, while staying in fp32. Use the
ready-made notebook `notebooks/kaggle_pipeline.ipynb`, or follow the steps below.

## Steps
1. Upload this zip on Kaggle: **Datasets → New Dataset → Upload** (Kaggle unzips it).
2. **New Notebook** (or import `notebooks/kaggle_pipeline.ipynb`), then right panel:
   - **Accelerator → GPU T4 x2** (or P100)
   - **Internet → On** (needed to download the Qwen model + ETHICS dataset)
   - **Add Input** → attach the dataset you just created.
3. Run the cells.

## Minimal cells
```python
!pip -q install sentence-transformers faiss-cpu
```
```python
import os, glob, shutil
src = os.path.dirname(os.path.dirname(glob.glob('/kaggle/input/**/src/run_all.py', recursive=True)[0]))
dst = '/kaggle/working/adl_rag'
shutil.rmtree(dst, ignore_errors=True); shutil.copytree(src, dst)
%cd /kaggle/working/adl_rag
```
```python
!python -m src.rag --build_index
!python -m src.run_all --quick --all --force   # smoke test
```
```python
!rm -rf outputs/
!python -m src.run_all --all                   # full run (drop --all for baseline+rag only)
!python scripts/make_figures.py
```

## Notes
- Outputs land in `/kaggle/working/adl_rag/{outputs,figures}`. Click **Save Version** to keep them.
- GPU sessions cap at ~9 h; checkpointing resumes the run if it disconnects.
- For ~2× more speed on GPU set `TORCH_DTYPE = "float16"` in `src/config.py` (results comparable).
