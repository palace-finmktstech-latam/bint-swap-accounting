Rule breakdown:

Curse-like rules (Cobertura='No' + Pata): 12
Termino-like rules (Cobertura='Sí' + Pata): 12
MTM rules (Cobertura='Sí' + no Pata): 48
Found 20 DESARME entries in interface file

Processing 2 DESARME trades for validation

KeyError: 'estrategia'
Traceback:
File "C:\Users\bencl\Proyectos\banco-internacional\test-contabilidad-swaps\main.py", line 336, in <module>
    desarme_results = validate_desarme_entries(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\bencl\Proyectos\banco-internacional\test-contabilidad-swaps\validators.py", line 2829, in validate_desarme_entries
    (mtm_rules['estrategia'] == previous_estrategia)
     ~~~~~~~~~^^^^^^^^^^^^^^
File "C:\Users\bencl\AppData\Local\Programs\Python\Python311\Lib\site-packages\pandas\core\frame.py", line 4102, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
File "C:\Users\bencl\AppData\Local\Programs\Python\Python311\Lib\site-packages\pandas\core\indexes\base.py", line 3812, in get_loc
    raise KeyError(key) from err