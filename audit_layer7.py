import torch
import plotly.express as px
from transformer_lens import HookedTransformer

def run_forensic_audit():
    """
    Performs a mechanistic interpretability audit on GPT-2 Layer 7
    to identify activation deltas between helpful and hostile prompts.
    """
    
    # 1. Load Model (CPU optimized for memory-constrained environments)
    print("Initializing model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")

    # 2. Define Audit Prompts
    # We compare a supportive 'world model' vs a hostile one.
    safe_text = "I want to help the community."
    unsafe_text = "I want to destroy the community."

    # 3. Extract Internal Activations (The 'Audit Trail')
    print("Extracting cache layers...")
    _, cache_safe = model.run_with_cache(safe_text)
    _, cache_unsafe = model.run_with_cache(unsafe_text)

    # 4. Calculate the Activation Delta (Delta = Unsafe - Safe)
    # Focusing on the MLP post-activation layer where 'reasoning' happens.
    diff = (cache_unsafe["blocks.7.mlp.hook_post"] - 
            cache_safe["blocks.7.mlp.hook_post"]).mean(dim=1).detach().squeeze()

    # 5. Professional Visualization
    # We use yaxis visible=False to prevent the 'sleeping' 0.5 label overlap.
    fig = px.imshow(
        diff[:50].unsqueeze(0).numpy(),
        labels=dict(x="Neuron Index", y="", color="Activation Delta"),
        title="<b>Forensic Audit: Safety Circuit Map (GPT-2 Layer 7)</b>",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    fig.update_layout(
        yaxis={'visible': False},  # Removes overlapping y-axis labels
        xaxis_title="Individual Neuron Index (Top 50 Neurons)",
        coloraxis_colorbar=dict(title="Delta"),
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="white",
        width=1000,
        height=350,
        title_x=0.5
    )

    print("Audit Complete. Displaying Map...")
    fig.show()

if __name__ == "__main__":
    run_forensic_audit()
