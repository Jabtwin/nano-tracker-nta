import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, x, y, width, height, text, facecolor='#e6f2ff', edgecolor='#0066cc'):
    box = patches.FancyBboxPatch((x, y), width, height,
                                 boxstyle="round,pad=0.1",
                                 facecolor=facecolor,
                                 edgecolor=edgecolor,
                                 linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=12, fontweight='bold', wrap=True)

def draw_arrow(ax, x_start, y_start, x_end, y_end):
    ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                arrowprops=dict(arrowstyle="->", color="black", lw=2, shrinkA=5, shrinkB=5))

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.5, 10.5)
ax.axis('off')

# Title
plt.title("Nano Particle Tracking Analysis (NTA) - Workflow", fontsize=16, fontweight='bold')

# Boxes
draw_box(ax, 3.0, 8.5, 4, 1, "1. Upload NTA Video\n(Streamlit UI)", facecolor='#e6f2ff')
draw_box(ax, 3.0, 6.5, 4, 1, "2. Image Processing\n(OpenCV: Grayscale, Blur, Threshold)", facecolor='#fff2e6', edgecolor='#cc6600')
draw_box(ax, 3.0, 4.5, 4, 1, "3. Particle Detection\n(Find Contours & Moments)", facecolor='#e6ffe6', edgecolor='#009900')
draw_box(ax, 3.0, 2.5, 4, 1, "4. Kalman Filter\n(Predict & Update trajectory)", facecolor='#ffe6e6', edgecolor='#cc0000')
draw_box(ax, 0.75, 0.5, 3.5, 1, "5. Physics Calculation\n(MSD & Stokes-Einstein)", facecolor='#f2e6ff', edgecolor='#6600cc')
draw_box(ax, 5.75, 0.5, 3.5, 1, "6. Report Generation\n(Matplotlib Charts)", facecolor='#f2e6ff', edgecolor='#6600cc')

# Arrows
draw_arrow(ax, 5, 8.5, 5, 7.5)
draw_arrow(ax, 5, 6.5, 5, 5.5)
draw_arrow(ax, 5, 4.5, 5, 3.5)
draw_arrow(ax, 5, 2.5, 2.5, 1.5)
draw_arrow(ax, 5, 2.5, 7.5, 1.5)

plt.tight_layout()
plt.savefig('images/architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print("Diagram saved to images/architecture.png")
