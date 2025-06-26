# Linux Dataset

This folder contains two text files extracted and processed from the `linux.tar.bz2` bug report dataset. Each file contains cleaned and categorized bug reports from the Linux kernel bug tracking system.

## 📂 Contents

- `positive_reports.txt`  
  Contains bug reports with a **positive resolution status**, such as:
  - `FIXED`
  - `VERIFIED`
  - `DUPLICATE`

- `negative_reports.txt`  
  Contains bug reports with a **non-positive or unresolved status**, such as:
  - `UNRESOLVED`
  - `WONTFIX`
  - `INVALID`, etc.

Each report includes:
- Bug ID
- Short Description
- Resolution status

## 🛠 Source and Processing

The original data was extracted from `linux.tar.bz2`, which contains XML-formatted bug reports. Python scripts were used to:

1. Extract the archive
2. Parse XML files
3. Classify bug reports into positive and negative
4. Save them as readable text files for easy access and further processing

## 🔍 Usage

These files can be used for:
- Training and evaluating bug classification models
- Studying real-world software issues and resolutions

## 📌 Note

- Some reports may have missing fields like `bug_id` or `short_desc`, handled gracefully during parsing.
- The files are plain text and human-readable.

