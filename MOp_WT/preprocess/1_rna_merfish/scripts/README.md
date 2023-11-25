
- Python scritps to preprocess decoded MERFISH data


- Brief description of the scripts as follows in the order of the jupyter notebooks:

   - RNA-MERFISH data is converted to [anndata_format](https://anndata.readthedocs.io/en/latest/) and then processed by [Scanpy](https://scanpy.readthedocs.io/en/stable/).

   - Initial prediction of cell types using prior MERFISH-data to refine the clustering results.

   - Manually rename (relabel) cell types and save the ouput for downstream analyses [postanalysis](../../../postanalysis/README.md) and 4DN data formatting [data_to_4dn](../../3_data_to_4dn/README.md)
