#!/usr/bin/env python3
"""Visualize conversation timeline with overlaps and interjections."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm

# Try to use Japanese font if available
try:
    # Common Japanese fonts on macOS
    japanese_fonts = ['Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Yu Gothic', 'Meiryo']
    available_fonts = [f.name for f in fm.findSystemFonts()]
    japanese_font = None
    for font in japanese_fonts:
        if any(font in available for available in available_fonts):
            japanese_font = font
            break
    if japanese_font:
        plt.rcParams['font.family'] = japanese_font
except:
    pass

def create_conversation_timeline():
    """Create a visual representation of conversation timeline."""
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    
    # Common settings
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(0, 30)
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlabel('Time (seconds) / 時間（秒）', fontsize=10)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Speaker 2\n話者2', 'Speaker 1\n話者1'])
        ax.grid(True, alpha=0.3, axis='x')
    
    # Example 1: Normal conversation with overlaps
    ax1.set_title('Example 1: Normal Conversation (Easy Sample) / 通常の会話（イージーサンプル）', fontsize=12, pad=10)
    
    # Speaker 1 segments
    segments1 = [
        (0, 3, 'Hello, how are you?'),
        (5.5, 8, 'I\'m doing great!'),
        (11, 14.5, 'What about the weather?'),
        (17, 19, 'Yes, exactly!'),
        (22, 25, 'See you tomorrow!')
    ]
    
    # Speaker 2 segments  
    segments2 = [
        (2.5, 5, 'Hi! I\'m fine.'),  # 0.5s overlap
        (7.5, 11.5, 'That\'s wonderful to hear'),  # 0.5s overlap
        (14, 17.5, 'It\'s quite sunny today'),  # 0.5s overlap
        (19.5, 22.5, 'Have a great day!'),  # 0.5s pause
        (24.5, 27, 'Goodbye!')  # 0.5s overlap
    ]
    
    # Draw segments
    for start, end, text in segments1:
        rect = FancyBboxPatch((start, 1.2), end-start, 0.6, 
                              boxstyle="round,pad=0.02", 
                              facecolor='lightblue', edgecolor='darkblue', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(start + (end-start)/2, 1.5, text, ha='center', va='center', fontsize=8)
    
    for start, end, text in segments2:
        rect = FancyBboxPatch((start, 0.2), end-start, 0.6,
                              boxstyle="round,pad=0.02",
                              facecolor='lightcoral', edgecolor='darkred', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(start + (end-start)/2, 0.5, text, ha='center', va='center', fontsize=8)
    
    # Add overlap indicators
    ax1.axvspan(2.5, 3, alpha=0.3, color='yellow', label='Overlap / 重なり')
    ax1.axvspan(7.5, 8, alpha=0.3, color='yellow')
    ax1.axvspan(14, 14.5, alpha=0.3, color='yellow')
    ax1.axvspan(24.5, 25, alpha=0.3, color='yellow')
    
    # Example 2: Conversation with interjections
    ax2.set_title('Example 2: Conversation with Interjections / 相槌を含む会話', fontsize=12, pad=10)
    
    # Speaker 1 segments (main)
    segments1_main = [
        (0, 4.5, 'Let me tell you about...'),
        (8, 13, 'It was really amazing when...'),
        (16, 21, 'And then what happened was...'),
        (24, 28, 'That\'s the whole story')
    ]
    
    # Speaker 2 segments (main + interjections)
    segments2_main = [
        (4, 8.5, 'Oh, please tell me more!'),
        (12.5, 16.5, 'That sounds interesting'),
        (20.5, 24.5, 'I can\'t believe it!')
    ]
    
    # Interjections (short responses)
    interjections = [
        (2, 2.5, 'Uh-huh', 2),  # Speaker 2 interjection
        (10, 10.4, 'Wow!', 2),   # Speaker 2 interjection
        (18.5, 19, 'Really?', 2), # Speaker 2 interjection
        (14.5, 15, 'Yeah', 1),   # Speaker 1 interjection
        (22.5, 23, 'Right', 1)   # Speaker 1 interjection
    ]
    
    # Draw main segments
    for start, end, text in segments1_main:
        rect = FancyBboxPatch((start, 1.2), end-start, 0.6,
                              boxstyle="round,pad=0.02",
                              facecolor='lightblue', edgecolor='darkblue', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(start + (end-start)/2, 1.5, text, ha='center', va='center', fontsize=8)
    
    for start, end, text in segments2_main:
        rect = FancyBboxPatch((start, 0.2), end-start, 0.6,
                              boxstyle="round,pad=0.02",
                              facecolor='lightcoral', edgecolor='darkred', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(start + (end-start)/2, 0.5, text, ha='center', va='center', fontsize=8)
    
    # Draw interjections
    for start, end, text, speaker in interjections:
        y_pos = 1.2 if speaker == 1 else 0.2
        rect = FancyBboxPatch((start, y_pos), end-start, 0.6,
                              boxstyle="round,pad=0.02",
                              facecolor='lightyellow', edgecolor='orange', 
                              linewidth=2, linestyle='dashed')
        ax2.add_patch(rect)
        ax2.text(start + (end-start)/2, y_pos + 0.3, text, 
                ha='center', va='center', fontsize=7, style='italic')
    
    # Example 3: Hard sample (similar speakers)
    ax3.set_title('Example 3: Hard Sample - Similar Speakers / ハードサンプル - 類似話者', fontsize=12, pad=10)
    
    # Both speakers have similar patterns (harder to separate)
    segments1_hard = [
        (0, 3.5, 'Good morning everyone'),
        (6, 9.5, 'Today we\'ll discuss...'),
        (12, 15, 'First point is...'),
        (18, 21.5, 'Second consideration...'),
        (24, 27, 'In conclusion...')
    ]
    
    segments2_hard = [
        (3, 6.5, 'Morning! Let\'s begin'),
        (9, 12.5, 'I agree with that'),
        (14.5, 18.5, 'Additionally, we should...'),
        (21, 24.5, 'That\'s a good point'),
        (26.5, 29, 'Thank you all')
    ]
    
    # Draw segments with similar colors (to show similarity)
    for start, end, text in segments1_hard:
        rect = FancyBboxPatch((start, 1.2), end-start, 0.6,
                              boxstyle="round,pad=0.02",
                              facecolor='#b3d9ff', edgecolor='#0066cc', linewidth=2)
        ax3.add_patch(rect)
        ax3.text(start + (end-start)/2, 1.5, text, ha='center', va='center', fontsize=8)
    
    for start, end, text in segments2_hard:
        rect = FancyBboxPatch((start, 0.2), end-start, 0.6,
                              boxstyle="round,pad=0.02",
                              facecolor='#99ccff', edgecolor='#0052cc', linewidth=2)
        ax3.add_patch(rect)
        ax3.text(start + (end-start)/2, 0.5, text, ha='center', va='center', fontsize=8)
    
    # Highlight overlaps
    ax3.axvspan(3, 3.5, alpha=0.3, color='yellow')
    ax3.axvspan(9, 9.5, alpha=0.3, color='yellow')
    ax3.axvspan(14.5, 15, alpha=0.3, color='yellow')
    ax3.axvspan(21, 21.5, alpha=0.3, color='yellow')
    ax3.axvspan(26.5, 27, alpha=0.3, color='yellow')
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor='lightblue', edgecolor='darkblue', label='Speaker 1 (Main) / 話者1（メイン）'),
        patches.Patch(facecolor='lightcoral', edgecolor='darkred', label='Speaker 2 (Main) / 話者2（メイン）'),
        patches.Patch(facecolor='lightyellow', edgecolor='orange', label='Interjection / 相槌'),
        patches.Patch(facecolor='yellow', alpha=0.3, label='Overlap Region / 重なり領域')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Add annotations
    ax1.text(15, -0.3, 'Natural turn-taking with 0-3s overlaps\n自然なターンテイキング（0-3秒の重なり）', 
             ha='center', fontsize=9, style='italic')
    ax2.text(15, -0.3, 'Short interjections during long utterances\n長い発話中の短い相槌', 
             ha='center', fontsize=9, style='italic')
    ax3.text(15, -0.3, 'Similar voice characteristics (harder to separate)\n類似した声質（分離が困難）', 
             ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig('conversation_timeline_visualization.png', dpi=300, bbox_inches='tight')
    plt.savefig('conversation_timeline_visualization.pdf', bbox_inches='tight')
    print("Saved visualization to conversation_timeline_visualization.png and .pdf")
    
    # Create a second figure showing the probability distributions
    fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Overlap distribution
    ax4.set_title('Overlap Duration Distribution\n重なり時間の分布', fontsize=12)
    overlap_durations = np.random.uniform(0, 3, 1000)
    ax4.hist(overlap_durations, bins=30, color='skyblue', edgecolor='darkblue', alpha=0.7)
    ax4.axvline(np.mean(overlap_durations), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(overlap_durations):.1f}s')
    ax4.set_xlabel('Overlap Duration (seconds) / 重なり時間（秒）')
    ax4.set_ylabel('Frequency / 頻度')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Interjection probability
    ax5.set_title('Interjection Probability by Utterance Duration\n発話時間による相槌確率', fontsize=12)
    durations = np.linspace(0, 5, 100)
    probabilities = np.minimum(0.6, durations * 0.2)
    ax5.plot(durations, probabilities, 'b-', linewidth=3)
    ax5.fill_between(durations, 0, probabilities, alpha=0.3)
    ax5.set_xlabel('Utterance Duration (seconds) / 発話時間（秒）')
    ax5.set_ylabel('Interjection Probability / 相槌確率')
    ax5.set_ylim(0, 0.7)
    ax5.grid(True, alpha=0.3)
    
    # Add annotations
    ax5.annotate('20% at 1s', xy=(1, 0.2), xytext=(1.5, 0.25),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    ax5.annotate('40% at 2s', xy=(2, 0.4), xytext=(2.5, 0.45),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    ax5.annotate('60% at 3s+', xy=(3, 0.6), xytext=(3.5, 0.65),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('conversation_statistics.png', dpi=300, bbox_inches='tight')
    plt.savefig('conversation_statistics.pdf', bbox_inches='tight')
    print("Saved statistics to conversation_statistics.png and .pdf")
    
    plt.show()


if __name__ == "__main__":
    create_conversation_timeline()