# Observation type ablations
# python run_ctf_agent.py --config configs/claude_opus4.yaml --headless
# python run_ctf_agent.py --config configs/claude_sonnet4.yaml --headless
# python run_ctf_agent.py --config configs/claude_sonnet3_5.yaml --headless
# python run_ctf_agent.py --config configs/claude_sonnet3_7.yaml --headless
# python run_ctf_agent.py --config configs/qwen2.5vl.yaml --headless

# python run_ctf_agent.py --config configs/som/claude_sonnet3_5.yaml --headless --observation_type som
# python run_ctf_agent.py --config configs/som/claude_opus4.yaml --headless --observation_type som
# python run_ctf_agent.py --config configs/som/claude_sonnet4.yaml --headless --observation_type som
# python run_ctf_agent.py --config configs/som/claude_sonnet3_7.yaml --headless --observation_type som
# python run_ctf_agent.py --config configs/som/qwen2.5vl.yaml --headless --observation_type som

# python run_ctf_agent.py --config configs/screenshot_a11y/claude_sonnet3_5.yaml --headless --observation_type screenshot_a11y_tree
# python run_ctf_agent.py --config configs/screenshot_a11y/claude_opus4.yaml --headless --observation_type screenshot_a11y_tree
# python run_ctf_agent.py --config configs/screenshot_a11y/claude_sonnet4.yaml --headless --observation_type screenshot_a11y_tree
# python run_ctf_agent.py --config configs/screenshot_a11y/claude_sonnet3_7.yaml --headless --observation_type screenshot_a11y_tree
# python run_ctf_agent.py --config configs/screenshot_a11y/qwen2.5vl.yaml --headless --observation_type screenshot_a11y_tree

# UI-TARS-1.5
# python run_ctf_agent.py --config configs/uitars1.5.yaml --headless

# Long trajectory experiments
python run_ctf_agent.py --config configs/long_claude_sonnet3_7.yaml --headless