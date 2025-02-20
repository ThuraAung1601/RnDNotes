Originally from https://github.com/sioaeko/NLLB_translator
<br>
I fixed some errors 
- AttributeError: 'NllbTokenizerFast' object has no attribute 'lang_code_to_id'
  - Fixed using tokenizer.convert_tokens_to_ids(LANG) e.g tokenizer.convert_tokens_to_ids("eng_Latn") instead of using lang_code_to_id
- File translation error: 'NamedString' object has no attribute 'file'
  - Extracts the file path (file.name) instead of trying to read it directly.
  - Uses open(file_path, 'r', encoding='utf-8') to manually read the file.
  - Uses os.path.basename(file_path) to prevent saving unwanted paths.

Run like this
```
python app.py
```
