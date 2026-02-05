import rahil

out = rahil.generate_lhs(Ninit=10, seed=42, output_dir="outputs")

print("Cache:", out.cache_dir)
print("Used Excel:", out.used_excel)
print("Used NetCDF:", out.used_base_nc)
print("NC written to:", out.param_output_dir)
print("Main run list:", out.main_run_file)
print("Param list:", out.param_list_file)
print(out.psets_df.head())