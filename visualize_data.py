import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import json
from datetime import datetime
import os
import plotly.express as px

# Global style overrides for dark theme & bold text
mpl.rcParams['font.family']      = 'DejaVu Sans'
mpl.rcParams['font.size']        = 12
mpl.rcParams['font.weight']      = 'bold'
mpl.rcParams['axes.titlesize']   = 20
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize']   = 14
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['text.color']       = 'white'
mpl.rcParams['axes.labelcolor']  = 'white'
mpl.rcParams['xtick.color']      = 'white'
mpl.rcParams['ytick.color']      = 'white'

# Color constants
FIG_BG      = "#1a3e1a"  # dark forest
AX_BG       = "#276627"  # lighter forest
LINE_COLOR  = 'white'
BOX_FILL    = "#5cb85c"  # bright green box fill
BOX_EDGE    = "#3e8e3e"  # dark green box edge

# Set style for better-looking plots
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", palette="Greens_d")

def get_db_connection():
    conn = sqlite3.connect('worldcup.db')
    return conn

def plot_goals_by_team():
    conn = get_db_connection()
    query = """
    SELECT t.name as team, COUNT(*) as goals
    FROM match_events me
    JOIN teams t ON me.acting_team_id = t.id
    WHERE me.event_type = 'shot' AND me.success = 1
    GROUP BY t.name
    ORDER BY goals DESC
    """
    df = pd.read_sql_query(query, conn)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    
    sns.barplot(data=df, x='team', y='goals', ax=ax, color=BOX_FILL, edgecolor=BOX_EDGE)
    ax.set_title('Goals Scored by Team')
    ax.set_xlabel('Team')
    ax.set_ylabel('Goals')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('goals_by_team.png', facecolor=FIG_BG, edgecolor='none')
    plt.close()
    conn.close()

def plot_bathroom_usage():
    conn = get_db_connection()
    query = """
    SELECT bathroom_name, COUNT(*) as usage_count
    FROM bathroom_usage
    GROUP BY bathroom_name
    """
    df = pd.read_sql_query(query, conn)
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    
    colors = [BOX_FILL, BOX_EDGE, "#4CAF50", "#81C784"]  # Green shades
    wedges, texts, autotexts = ax.pie(
        df['usage_count'], 
        labels=df['bathroom_name'], 
        autopct='%1.1f%%',
        colors=colors, 
        textprops={'color': 'white', 'weight': 'bold', 'fontsize': 16},
        startangle=90
    )
    # Set all label text to white, bold, and larger
    for text in texts:
        text.set_color('white')
        text.set_fontsize(16)
        text.set_weight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(18)
        autotext.set_weight('bold')
    ax.set_title('Bathroom Usage Distribution', color='white', fontsize=26, weight='bold', pad=30)
    plt.tight_layout()
    plt.savefig('bathroom_usage.png', facecolor=FIG_BG, edgecolor='none')
    plt.close()
    conn.close()

def plot_popular_food_items():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT items_json FROM orders", conn)
    item_counts = {}
    for items_json_str in df['items_json']:
        try:
            items = json.loads(items_json_str)
            for item in items:
                item_counts[item] = item_counts.get(item, 0) + 1
        except Exception:
            continue
    items_df = pd.DataFrame(list(item_counts.items()), columns=['item_name', 'order_count'])
    items_df = items_df.sort_values('order_count', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=items_df, x='item_name', y='order_count')
    plt.title('Most Popular Food Items')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('popular_food_items.png')
    plt.close()
    conn.close()

def plot_match_timeline():
    conn = get_db_connection()
    query = """
    SELECT me.match_id, me.event_type, me.event_minute, t.name as team
    FROM match_events me
    JOIN teams t ON me.acting_team_id = t.id
    WHERE me.event_type IN ('shot', 'corner', 'foul')
    """
    df = pd.read_sql_query(query, conn)
    plt.figure(figsize=(15, 8))
    sns.scatterplot(data=df, x='event_minute', y='match_id', hue='event_type', style='team', alpha=0.7)
    plt.title('Match Events Timeline')
    plt.xlabel('Event Time (minutes)')
    plt.ylabel('Match ID')
    plt.tight_layout()
    plt.savefig('match_timeline.png')
    plt.close()
    conn.close()

def plot_ranking_vs_performance():
    conn = get_db_connection()
    query = """
    SELECT t.name as team, t.ranking, COUNT(DISTINCT m.id) as matches_played,
           SUM(CASE WHEN me.event_type = 'shot' AND me.success = 1 THEN 1 ELSE 0 END) as total_goals
    FROM teams t
    LEFT JOIN matches m ON t.id = m.team1_id OR t.id = m.team2_id
    LEFT JOIN match_events me ON m.id = me.match_id AND me.acting_team_id = t.id
    GROUP BY t.name, t.ranking
    HAVING matches_played > 0
    """
    df = pd.read_sql_query(query, conn)
    df['goals_per_match'] = df['total_goals'] / df['matches_played']
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='ranking', y='goals_per_match')
    plt.title('Team Performance: Goals per Match vs. Ranking')
    plt.xlabel('FIFA Ranking (lower is better)')
    plt.ylabel('Goals per Match')
    for i, row in df.iterrows():
        plt.annotate(row['team'], (row['ranking'], row['goals_per_match']))
    plt.tight_layout()
    plt.savefig('ranking_vs_performance.png')
    plt.close()
    conn.close()

def plot_team_event_heatmap():
    conn = get_db_connection()
    query = """
    SELECT 
        t.name as team,
        me.event_type,
        COUNT(*) as event_count
    FROM match_events me
    JOIN teams t ON me.acting_team_id = t.id
    WHERE me.event_type IN ('shot', 'corner', 'foul', 'tackle', 'dribble')
    GROUP BY t.name, me.event_type
    """
    df = pd.read_sql_query(query, conn)
    
    # Pivot the data to create a matrix for the heatmap
    pivot_df = df.pivot(index='team', columns='event_type', values='event_count')
    
    # Normalize the data by dividing each row by its sum to show relative proportions
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, 
                annot=True, 
                fmt='.2f', 
                cmap='Greens',
                cbar_kws={'label': 'Proportion of Events'})
    plt.title('Team Play Style Analysis\n(Proportion of Different Event Types)')
    plt.xlabel('Event Type')
    plt.ylabel('Team')
    plt.tight_layout()
    plt.savefig('team_event_heatmap.png')
    plt.close()
    conn.close()

def plot_event_correlations():
    """Create a correlation heatmap between different match events"""
    conn = get_db_connection()
    query = """
    SELECT 
        m.id as match_id,
        t.name as team,
        me.event_type,
        COUNT(*) as event_count
    FROM matches m
    JOIN match_events me ON m.id = me.match_id
    JOIN teams t ON me.acting_team_id = t.id
    WHERE me.event_type IN ('shot', 'corner', 'foul', 'tackle', 'dribble')
    GROUP BY m.id, t.name, me.event_type
    """
    df = pd.read_sql_query(query, conn)
    
    # Pivot the data to create a correlation matrix
    pivot_df = df.pivot_table(
        index=['match_id', 'team'],
        columns='event_type',
        values='event_count',
        fill_value=0
    ).reset_index()
    
    # Calculate correlation matrix
    corr_matrix = pivot_df.drop(['match_id', 'team'], axis=1).corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='RdYlGn',
                center=0,
                fmt='.2f',
                square=True)
    plt.title('Correlation Between Different Match Events')
    plt.tight_layout()
    plt.savefig('event_correlations.png')
    plt.close()
    conn.close()

def plot_spectator_entries():
    """Plot the cumulative number of spectator entries over time"""
    conn = get_db_connection()
    query = """
    SELECT 
        m.match_datetime,
        COUNT(*) as entry_count
    FROM spectator_entries se
    JOIN matches m ON se.match_id = m.id
    GROUP BY m.match_datetime
    ORDER BY m.match_datetime
    """
    df = pd.read_sql_query(query, conn)
    df['match_datetime'] = pd.to_datetime(df['match_datetime'])
    df['cumulative_entries'] = df['entry_count'].cumsum()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['match_datetime'], df['cumulative_entries'], marker='o')
    plt.title('Cumulative Spectator Entries Over Time')
    plt.xlabel('Match Date')
    plt.ylabel('Total Spectators')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('spectator_entries.png')
    plt.close()
    conn.close()

def plot_top_scorers():
    conn = get_db_connection()
    query = """
    SELECT player_name, COUNT(*) as goals
    FROM match_events
    WHERE event_type = 'shot' AND success = 1 AND player_name != ''
    GROUP BY player_name
    ORDER BY goals DESC
    LIMIT 10
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    podium_colors = ['gold', 'silver', '#cd7f32'] + [BOX_FILL] * (len(df) - 3)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)
    
    bars = ax.bar(df['player_name'], df['goals'], color=podium_colors, edgecolor=BOX_EDGE, linewidth=1.5)
    
    ax.set_title('Top 10 Goal Scorers', pad=20, color='white', weight='bold')
    ax.set_xlabel('Player', color='white', weight='bold')
    ax.set_ylabel('Goals', color='white', weight='bold')
    plt.xticks(rotation=45, ha='right', color='white', weight='bold')
    plt.yticks(color='white', weight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}',
                ha='center', va='bottom',
                color='white', weight='bold')
    
    plt.tight_layout()
    plt.savefig('top_scorers_podium.png', facecolor=FIG_BG, edgecolor='none', dpi=300)
    plt.close()

def plot_goalkeeper_saves_podium():
    # Load starting_lineups from team_data.json
    with open('team_data.json', 'r', encoding='utf-8') as f:
        team_data = json.load(f)
    starting_lineups = team_data['starting_lineups']

    conn = get_db_connection()
    query = """
    SELECT t.name as team,
           SUM(CASE WHEN me.acting_team_id != t.id AND me.event_type = 'shot' THEN 1 ELSE 0 END) as shots_faced,
           SUM(CASE WHEN me.acting_team_id != t.id AND me.event_type = 'shot' AND me.success = 1 THEN 1 ELSE 0 END) as goals_conceded
    FROM matches m
    JOIN teams t ON t.id IN (m.team1_id, m.team2_id)
    LEFT JOIN match_events me ON me.match_id = m.id
    GROUP BY t.name
    """
    df = pd.read_sql_query(query, conn)
    df['goalkeeper'] = df['team'].apply(lambda team: starting_lineups.get(team, ['Unknown'])[0])
    df['saves'] = df['shots_faced'] - df['goals_conceded']
    df = df.sort_values('saves', ascending=False).head(10)
    conn.close()

    podium_colors = ['gold', 'silver', '#cd7f32'] + [BOX_FILL] * (len(df) - 3)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(AX_BG)

    bars = ax.bar(df['goalkeeper'], df['saves'], color=podium_colors, edgecolor=BOX_EDGE, linewidth=1.5)

    ax.set_title('Top 10 Goalkeepers by Estimated Saves', pad=20, color='white', weight='bold')
    ax.set_xlabel('Goalkeeper', color='white', weight='bold')
    ax.set_ylabel('Estimated Saves', color='white', weight='bold')
    plt.xticks(rotation=45, ha='right', color='white', weight='bold')
    plt.yticks(color='white', weight='bold')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}',
                ha='center', va='bottom',
                color='white', weight='bold')

    ax.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('goalkeeper_saves_podium.png', facecolor=FIG_BG, edgecolor='none', dpi=300)
    plt.close()

def plot_cafeteria_sunburst():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT items_json FROM orders", conn)
    conn.close()
    # Parse items and count number of items per order
    df['items'] = df['items_json'].apply(json.loads)
    df['num_items'] = df['items'].apply(len)
    exploded = df.explode('items')
    # Sunburst: Level 1 = num_items, Level 2 = item name
    fig = px.sunburst(
        exploded,
        path=['num_items', 'items'],
        title='Cafeteria Orders: Items per Order and Popularity',
        color='num_items',
        color_continuous_scale='greens',
        width=700,
        height=500
    )
    fig.write_image('cafeteria_sunburst.png')

def plot_bathroom_boxen():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT bathroom_name, use_duration FROM bathroom_usage", conn)
    conn.close()
    plt.figure(figsize=(12, 6))
    sns.boxenplot(data=df, x='bathroom_name', y='use_duration', palette='Greens')
    plt.title('Distribution of Bathroom Usage Duration by Bathroom')
    plt.xlabel('Bathroom')
    plt.ylabel('Usage Duration (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('bathroom_boxen.png')
    plt.close()

def main():
    print("Generating visualizations...")
    plot_goals_by_team()
    plot_bathroom_usage()
    plot_popular_food_items()
    plot_ranking_vs_performance()
    plot_team_event_heatmap()
    plot_event_correlations()
    plot_spectator_entries()
    plot_top_scorers()
    plot_goalkeeper_saves_podium()
    plot_cafeteria_sunburst()
    plot_bathroom_boxen()
    print("Visualizations complete! Check the generated PNG files.")

if __name__ == "__main__":
    main() 