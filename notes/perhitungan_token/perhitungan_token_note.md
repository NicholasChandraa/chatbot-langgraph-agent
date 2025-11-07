Perhitungan Tokens Langchain:

output_tokens = text_output + reasoning_output
total_tokens = input_tokens + output_tokens

cache_read_token = sudah termasuk di dalam input_tokens

Formula:
input_tokens = fresh_input + cached_input