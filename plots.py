import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_cumulative_profit(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="step", y="cumulative_profit", hue="Agent", linewidth=2)
    plt.title("Cumulative Profit Comparison")
    plt.ylabel("Profit ($)")
    plt.xlabel("Days")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_cumulative_profit.png"))
    plt.close()

def plot_inventory_dynamics(df, output_dir, zoom_window=60):
    # Find the middle of the episode to zoom in
    unique_agents = df['Agent'].unique()
    if len(unique_agents) == 0: return
    
    # Calculate approx middle step
    total_steps = len(df) // len(unique_agents)
    mid_point = total_steps // 2
    
    subset = df[(df['step'] > mid_point) & (df['step'] < mid_point + zoom_window)]
    
    plt.figure(figsize=(12, 6))
    
    # Draw Demand background (using the first agent's demand data)
    first_agent_data = subset[subset['Agent'] == unique_agents[0]]
    plt.fill_between(subset['step'].unique(), first_agent_data['demand'], 
                     color='gray', alpha=0.2, label='Demand')
    
    sns.lineplot(data=subset, x="step", y="inventory", hue="Agent", linewidth=2)
    plt.axhline(0, color='red', linestyle='--')
    
    plt.title(f"Inventory Levels vs Demand ({zoom_window} Day Zoom)")
    plt.ylabel("Units")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_inventory_dynamics.png"))
    plt.close()

def plot_action_distribution(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x="action", hue="Agent", fill=True, common_norm=False, alpha=0.3)
    plt.title("Action Distribution (Order Sizes)")
    plt.xlabel("Order Quantity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_action_distribution.png"))
    plt.close()

def plot_efficiency_frontier(metrics_df, output_dir):
    plt.figure(figsize=(8, 8))
    
    sns.scatterplot(
        data=metrics_df, 
        x="Avg Inventory", 
        y="Service Level (%)", 
        hue="Agent", 
        style="Agent", 
        s=200
    )
    
    for i, row in metrics_df.iterrows():
        plt.text(
            row['Avg Inventory'], 
            row['Service Level (%)'] + 0.2, 
            f"{row['Agent']}\n${row['Total Profit']:,.0f}", 
            fontsize=9
        )
        
    plt.title("Efficiency Frontier (Cost vs Service)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "4_efficiency_frontier.png"))
    plt.close()

def generate_plots(full_df, metrics_df, output_dir="results"):
    """
    Main function to generate all plots at once.
    """
    print("Generating plots...")
    
    sns.set_theme(style="whitegrid")
    
    plot_cumulative_profit(full_df, output_dir)
    plot_inventory_dynamics(full_df, output_dir)
    plot_action_distribution(full_df, output_dir)
    plot_efficiency_frontier(metrics_df, output_dir)
    
    print(f"Plots saved to {output_dir}/")