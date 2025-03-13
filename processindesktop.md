downloading computer

Windows .venv
The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.
0it [00:00, ?it/s]
WARNING: Failed to find MSVC.
cuda:0
config.json: 100%|████████████████████████████████████████████████████████████████████████████| 684/684 [00:00<?, ?B/s]
C:\Users\24007516\GitHub\YuE-for-windows\.venv\lib\site-packages\huggingface_hub\file_download.py:140: UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\24007516\GitHub\YuE-for-windows\huggingface\hub\models--m-a-p--YuE-s1-7B-anneal-en-cot. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)
model.safetensors.index.json: 100%|███████████████████████████████████████████████████████| 23.9k/23.9k [00:00<?, ?B/s]
Downloading shards:   0%|                                                                        | 0/3 [00:00<?, ?it/s]
model-00001-of-00003.safetensors:  74%|████████████████████████████████▌           | 3.64G/4.92G [00:21<00:05, 244MB/s]




yue environment

(base) PS C:\Users\24007516\GitHub\YuE-for-windows> conda activate yue
(yue) PS C:\Users\24007516\GitHub\YuE-for-windows> python -c "import sys; print(sys.executable)"
C:\ProgramData\anaconda3\envs\yue\python.exe
(yue) PS C:\Users\24007516\GitHub\YuE-for-windows>




module load cuda11.8/toolkit/11.8.0





