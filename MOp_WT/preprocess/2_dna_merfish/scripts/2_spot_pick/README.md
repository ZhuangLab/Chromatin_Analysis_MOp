

- Python scritps to pick candidate DNA spots to construct chromatin fibers.


- Brief description of the scripts as follows in the order of the jupyter notebooks:

   - 1. Export decoded data for corresponding replicate into a format that is compatible for spot picking by Jie;

     - For example, obtain exported decoded data from [20220713-NewPostAnalysis3_Decoding](../1_spot_preprocess/20220713-NewPostAnalysis3_Decoding.ipynb)

   - 2. Pick spots by the modified [Jie](https://github.com/cosmosyw/jie/tree/0f797efcc52c1e9822a3cd03d8980cba9315b468).


   - 3. Sort picked spots by function "sort_spots_by_fiber_density", which can also be found in the same [sort_spot_notebook](3_sort_jie_spots.ipynb) if not in the modified [Jie](https://github.com/cosmosyw/jie/tree/0f797efcc52c1e9822a3cd03d8980cba9315b468).


   - 4. Summarize the sorted spots into dictionary, etc., for downstream [postanalysis](../../../../postanalysis/README.md)