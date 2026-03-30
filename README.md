# 動畫語音前處理工具

這個專案目前提供一條簡單可用的流程，用來把動畫或音訊素材做成人工可審核的語音資料：

1. 先用 Demucs 做人聲分離
2. 把 `vocals.wav` 切成較短片段
3. 逐段人工標記 `yes` / `no` / `maybe`

這份 README 目前只說明根目錄與 `audio_separation` 相關功能，不包含 `voice` 目錄內的內容。

## 專案結構

```text
.
├── README.md
├── requirements.txt
├── separate_audio.py
├── split_vocals.py
├── review_segments.py
└── audio_separation
    ├── __init__.py
    ├── cli.py
    ├── ffmpeg_utils.py
    ├── io_utils.py
    ├── logging_utils.py
    ├── review_cli.py
    ├── segmenter.py
    ├── split_cli.py
    └── separators
        ├── __init__.py
        ├── base.py
        └── demucs_separator.py
```

## 環境需求

- Python 3.10 以上
- `ffmpeg`
- `pip install -r requirements.txt`

安裝依賴：

```powershell
pip install -r requirements.txt
```

目前 `requirements.txt` 內容：

- `demucs`
- `numpy`
- `soundfile`

注意：

- Demucs 第一次執行時可能會下載模型權重
- 若要使用 GPU，需要安裝 CUDA 版 PyTorch
- `ffmpeg` 必須存在於 `PATH` 中

## 安裝 ffmpeg

請先確認：

```powershell
ffmpeg -version
```

如果系統沒有 `ffmpeg`，Windows 可用：

```powershell
winget install Gyan.FFmpeg
```

安裝後請重新開啟終端，再確認 `ffmpeg -version` 可正常執行。

## 支援輸入格式

影片格式：

- `.mp4`
- `.mkv`
- `.mov`
- `.avi`
- `.webm`

音訊格式：

- `.wav`
- `.mp3`
- `.flac`
- `.m4a`
- `.aac`
- `.ogg`

## 1. 人聲分離

入口腳本：

- [separate_audio.py](/C:/Users/lee92/AI_Chino/separate_audio.py)

主要功能：

- 驗證輸入檔案
- 用 `ffmpeg` 把輸入轉成標準 WAV
- 呼叫 Demucs 進行 source separation
- 固定輸出：
  - `vocals.wav`
  - `accompaniment.wav`
  - `metadata.json`

基本用法：

```powershell
python .\separate_audio.py --input ".\input.mp3" --output_dir ".\outputs\wav_data"
```

使用 GPU：

```powershell
python .\separate_audio.py --input ".\input.mp3" --output_dir ".\outputs\wav_data" --device cuda
```

指定模型：

```powershell
python .\separate_audio.py --input ".\input.mp3" --output_dir ".\outputs\wav_data" --device cuda --model htdemucs_ft
```

常用參數：

- `--input`: 輸入音訊或影片檔
- `--output_dir`: 輸出資料夾
- `--separator`: 目前只支援 `demucs`
- `--model`: Demucs 模型名稱，預設 `htdemucs`
- `--device`: 執行裝置，例如 `cpu` 或 `cuda`
- `--sample_rate`: 預設 `44100`
- `--channels`: 預設 `2`

### 輸出內容

`output_dir` 內會產生：

- `vocals.wav`
  - 主要給後續切分、人工篩選、字幕對齊或訓練前處理使用
- `accompaniment.wav`
  - 非人聲部分
- `metadata.json`
  - 記錄這次執行的輸入、模型、參數與成功狀態

`metadata.json` 主要欄位包含：

- `input_path`
- `extracted_wav_path`
- `output_vocals_path`
- `output_accompaniment_path`
- `model_name`
- `sample_rate`
- `duration_seconds`
- `processing_time_seconds`
- `success`
- `error_message`
- `command_args`

## 2. 切分語音片段

入口腳本：

- [split_vocals.py](/C:/Users/lee92/AI_Chino/split_vocals.py)

主要功能：

- 讀取 `vocals.wav`
- 使用能量式規則偵測有人聲的區段
- 自動補短靜音
- 把過長片段再切短
- 輸出方便人工審核的小片段與 metadata

基本用法：

```powershell
python .\split_vocals.py --input ".\outputs\wav_data\vocals.wav" --output_dir ".\outputs\wav_segments"
```

常用參數：

- `--input`: 要切分的音訊檔，通常是 `vocals.wav`
- `--output_dir`: 切分輸出資料夾
- `--min_duration_ms`: 最短片段長度，預設 `800`
- `--max_duration_ms`: 最長片段長度，預設 `8000`
- `--pad_ms`: 每段前後補的長度，預設 `120`
- `--frame_ms`: 分析 frame 大小，預設 `30`
- `--hop_ms`: 分析 hop 大小，預設 `10`
- `--max_silence_ms`: 會被視為可橋接短靜音的長度，預設 `250`
- `--threshold_db`: 手動指定切分門檻；不填則自動估算

範例：

```powershell
python .\split_vocals.py `
  --input ".\outputs\wav_data\vocals.wav" `
  --output_dir ".\outputs\wav_segments" `
  --min_duration_ms 800 `
  --max_duration_ms 8000 `
  --pad_ms 120 `
  --max_silence_ms 250
```

### 輸出內容

`output_dir` 內會產生：

- `segments\0001.wav`, `0002.wav`, ...
- `segments.csv`
- `segments.json`

`segments.csv` 欄位：

- `segment_id`
- `file_name`
- `start_seconds`
- `end_seconds`
- `duration_seconds`
- `peak_dbfs`
- `rms_dbfs`
- `keep`
- `note`

說明：

- `keep` 目前留給人工標註使用
- `note` 可自行填 `bgm`、`overlap`、`distorted` 等備註

### 切分調整建議

如果切得太碎：

- 把 `--min_duration_ms` 調大，例如 `1200`
- 把 `--max_silence_ms` 調大，例如 `350`

如果切得太鬆：

- 把 `--max_silence_ms` 調小，例如 `150`
- 或手動指定 `--threshold_db`

## 3. 互動式人工標註

入口腳本：

- [review_segments.py](/C:/Users/lee92/AI_Chino/review_segments.py)

主要功能：

- 一段一段播放 `segments/*.wav`
- 在終端顯示目前檔名與長度
- 用鍵盤快速標記 `yes` / `no` / `maybe`
- 標記結果直接寫回 `segments.csv`
- 可從指定檔名續跑

基本用法：

```powershell
python .\review_segments.py --output_dir ".\outputs\wav_segments"
```

從指定檔案續跑：

```powershell
python .\review_segments.py --output_dir ".\outputs\wav_segments" --start 0042.wav
```

操作鍵：

- `y`: 標記為 `yes`
- `n`: 標記為 `no`
- `m`: 標記為 `maybe`
- `r`: 重播目前片段
- `q`: 停止本次工作

目前畫面會顯示類似：

```text
[42/255] Playing 0042.wav (2.37s)
```

行為說明：

- 每個 WAV 都必須按下 `y`、`n` 或 `m` 其中一個，才會進到下一段
- 若按 `r`，會重播目前片段
- 若按 `q`，會立即停止，已完成的標註會保留在 `segments.csv`
- 若沒有指定 `--start`，工具會優先從第一個尚未標記 `keep` 的片段開始

## 建議工作流程

針對動畫角色語音資料，建議流程如下：

1. 先用 `separate_audio.py` 將原始素材分離出 `vocals.wav`
2. 用 `split_vocals.py` 把 `vocals.wav` 切成短片段
3. 用 `review_segments.py` 逐段標記 `yes` / `no` / `maybe`
4. 後續再把 `yes` 的片段整理成訓練資料集

## 常見問題

### ffmpeg 找不到

若出現類似錯誤：

- `ffmpeg was not found in PATH`

請先安裝 ffmpeg，並確認：

```powershell
ffmpeg -version
```

### 輸入檔不是有效音訊

如果 `ffmpeg` 回報：

- `Invalid data found when processing input`

請先確認輸入檔是真正的音訊或影片檔，而不是下載失敗後留下的 HTML 錯誤頁。

### GPU 無法使用

若 `torch.cuda.is_available()` 為 `False`，代表目前 Python 環境可能裝的是 CPU 版 PyTorch。需要改安裝 CUDA 版套件後，才能用：

```powershell
--device cuda
```

### Demucs 模型效果不理想

可以試著切換模型，例如：

```powershell
python .\separate_audio.py --input ".\input.mp3" --output_dir ".\outputs\wav_data" --device cuda --model htdemucs_ft
```

如果想使用 `mdx_extra_q`，Windows 上通常還需要額外處理 `diffq` 與編譯環境問題。

## 音檔標註進度
(87/255)
