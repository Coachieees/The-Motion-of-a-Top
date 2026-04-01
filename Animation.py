import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os

# --- 1. ASK USER FOR FILE & LOAD DATA ---
# Prompt the user to type the file name
filename = "Data/case_2_1.txt"

# Check if the file actually exists before trying to load it
if not os.path.isfile(filename):
    print(f"Error: Could not find the file '{filename}'. Check the spelling!")
    sys.exit()

# Load the chosen file
data = np.loadtxt(filename, skiprows=1, delimiter=' ')
t_data = data[:, 0]
psi_data = data[:, 1]
the_data = data[:, 2]
phi_data = data[:, 3]

# --- 2. CONSTRUCT THE 3D TOP GEOMETRY ---
points = []
points.extend([[0,0,0], [0,0,1.5], [np.nan, np.nan, np.nan]])

r = 0.5
h = 1.0
theta_circ = np.linspace(0, 2*np.pi, 40)
for tc in theta_circ:
    points.append([r*np.cos(tc), r*np.sin(tc), h])
points.append([np.nan, np.nan, np.nan])

for tc in np.linspace(0, 2*np.pi, 8, endpoint=False):
    x, y = r*np.cos(tc), r*np.sin(tc)
    points.extend([[0,0,0], [x,y,h], [np.nan, np.nan, np.nan]]) 
    points.extend([[x,y,h], [0,0,1.5], [np.nan, np.nan, np.nan]]) 

geom_local = np.array(points).T 

# --- 3. ROTATION MATRIX FUNCTION ---
def get_rotation_matrix(psi, theta, phi):
    c1, s1 = np.cos(phi), np.sin(phi)
    R_phi = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    
    c2, s2 = np.cos(theta), np.sin(theta)
    R_theta = np.array([[1, 0, 0], [0, c2, -s2], [0, s2, c2]])
    
    c3, s3 = np.cos(psi), np.sin(psi)
    R_psi = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])
    
    return R_psi @ R_theta @ R_phi

# --- 4. SETUP THE ANIMATION FIGURE ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

bound = 1.5
ax.set_xlim([-bound, bound])
ax.set_ylim([-bound, bound])
ax.set_zlim([0, bound])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (Vertical)')
# Update the title dynamically based on the file loaded!
ax.set_title(f"3D Animation: {filename}")

top_lines, = ax.plot([], [], [], color='black', lw=1.5)
trail_lines, = ax.plot([], [], [], color='blue', lw=2, alpha=0.8)

trail_x, trail_y, trail_z = [], [], []

# --- 5. THE ANIMATION LOOP ---
def update(frame_index):
    psi = psi_data[frame_index]
    theta = the_data[frame_index]
    phi = phi_data[frame_index]
    
    R = get_rotation_matrix(psi, theta, phi)
    geom_global = R @ geom_local
    
    top_lines.set_data(geom_global[0, :], geom_global[1, :])
    top_lines.set_3d_properties(geom_global[2, :])
    
    center_global = R @ np.array([0, 0, 1.0])
    trail_x.append(center_global[0])
    trail_y.append(center_global[1])
    trail_z.append(center_global[2])
    
    if len(trail_x) > 300:
        trail_x.pop(0); trail_y.pop(0); trail_z.pop(0)
        
    trail_lines.set_data(trail_x, trail_y)
    trail_lines.set_3d_properties(trail_z)
    
    return top_lines, trail_lines

ani = animation.FuncAnimation(fig, update, frames=range(0, len(t_data), 5), 
                              interval=20, blit=False)

plt.show()