from datasets import load_dataset

dataset = load_dataset("shunk031/CAMERA", name="without-lp-images")

print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['asset_id', 'kw', 'lp_meta_description', 'title_org', 'title_ne1', 'title_ne2', 'title_ne3', 'domain', 'parsed_full_text_annotation'],
#         num_rows: 12395
#     })
#     validation: Dataset({
#         features: ['asset_id', 'kw', 'lp_meta_description', 'title_org', 'title_ne1', 'title_ne2', 'title_ne3', 'domain', 'parsed_full_text_annotation'],
#         num_rows: 3098
#     })
#     test: Dataset({
#         features: ['asset_id', 'kw', 'lp_meta_description', 'title_org', 'title_ne1', 'title_ne2', 'title_ne3', 'domain', 'parsed_full_text_annotation'],
#         num_rows: 872
#     })
# })
# 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
# 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