import rahil

out = rahil.generate_lhs(Ninit=150, seed=42, output_dir="outputs")

print(out.param_output_dir)
print(out.main_run_file)
print(out.param_list_file)
print(out.psets_df.head())

