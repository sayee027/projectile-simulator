import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import base64
from io import BytesIO
import colorsys

# Physics Constants
G = 9.81  # m/s²
AIR_DENSITY = 1.225  # kg/m³ at sea level

def hsl_to_hex(h, s, l):
    """Convert HSL values to hex color code"""
    r, g, b = colorsys.hls_to_rgb(h/360, l/100, s/100)
    return f"#{round(r*255):02x}{round(g*255):02x}{round(b*255):02x}"

def get_air_density(altitude):
    """Calculate air density based on altitude using barometric formula"""
    # Constants
    rho_0 = 1.225  # kg/m³ at sea level
    T_0 = 288.15   # K at sea level
    L = 0.0065     # K/m temperature lapse rate
    M = 0.0289644  # kg/mol molar mass of dry air
    R = 8.31447    # J/(mol·K) universal gas constant
    g = 9.81       # m/s² gravitational acceleration
    
    # Barometric formula
    temperature = T_0 - L * altitude
    exponent = g * M / (R * L)
    rho = rho_0 * (temperature / T_0) ** (exponent - 1)
    
    return rho

def calculate_drag_coefficient(reynolds_number, projectile_type='sphere'):
    """Calculate drag coefficient based on Reynolds number and projectile type"""
    if projectile_type == 'sphere':
        if reynolds_number < 0.5:
            return 24 / reynolds_number
        elif reynolds_number < 1000:
            return 24 / reynolds_number * (1 + 0.15 * reynolds_number**0.687)
        elif reynolds_number < 2e5:
            return 0.47
        else:
            return 0.2  # Supercritical flow
    elif projectile_type == 'bullet':
        return 0.295  # Typical value for bullets
    elif projectile_type == 'ping_pong':
        return 0.4  # Approximate value for ping pong balls
    else:
        return 0.47  # Default value for spheres

def simulate_projectile_advanced(
    launch_speed, launch_angle, mass, diameter,
    drag_model='constant', projectile_type='sphere',
    spin_rate=0, coriolis_enabled=False, latitude=45,
    dt=0.01, max_time=100,  # Increased max_time to prevent cutoffs
    wind_speed=0.0, wind_angle=0.0, launch_height=0.0
):
    """Advanced projectile motion simulation with multiple drag models and effects"""
    # Convert angles to radians
    theta = np.radians(launch_angle)
    wind_theta = np.radians(wind_angle)
    
    # Wind components
    wind_x = wind_speed * np.cos(wind_theta)
    wind_y = wind_speed * np.sin(wind_theta)
    
    # Initial conditions
    vx = launch_speed * np.cos(theta)
    vy = launch_speed * np.sin(theta)
    x, y, t = 0, launch_height, 0
    
    # Pre-calculate constants
    area = np.pi * (diameter/2)**2
    
    # Coriolis parameter (2Ω sin φ)
    omega = 7.2921e-5  # Earth's rotation rate in rad/s
    coriolis_param = 2 * omega * np.sin(np.radians(latitude)) if coriolis_enabled else 0
    
    # Storage for results
    times, x_vals, y_vals = [], [], []
    vx_vals, vy_vals = [], []
    
    # Dynamic viscosity of air (kg/(m·s))
    mu = 1.81e-5
    
    while y >= 0 and t < max_time:
        times.append(t)
        x_vals.append(x)
        y_vals.append(y)
        vx_vals.append(vx)
        vy_vals.append(vy)
        
        # Current air density based on altitude
        rho = get_air_density(y)
        
        # Relative velocity including wind
        vx_rel = vx - wind_x
        vy_rel = vy - wind_y
        speed_rel = np.sqrt(vx_rel**2 + vy_rel**2)
        
        # Safety check for zero speed
        if speed_rel < 1e-10:
            speed_rel = 1e-10  # Prevent division by zero
            
        # Reynolds number
        reynolds = rho * speed_rel * diameter / mu
        
        # Drag force calculation based on selected model
        if drag_model == 'constant':
            # Constant drag coefficient
            k = 0.5 * rho * 0.47 * area
            drag_x = -k * speed_rel * vx_rel
            drag_y = -k * speed_rel * vy_rel
        elif drag_model == 'variable_cd':
            # Variable drag coefficient based on Reynolds number
            cd = calculate_drag_coefficient(reynolds, projectile_type)
            k = 0.5 * rho * cd * area
            drag_x = -k * speed_rel * vx_rel
            drag_y = -k * speed_rel * vy_rel
        elif drag_model == 'linear':
            # Linear drag model (F = -bv)
            b = 0.1  # Damping coefficient
            drag_x = -b * vx_rel
            drag_y = -b * vy_rel
        elif drag_model == 'quadratic':
            # Quadratic drag model
            cd = calculate_drag_coefficient(reynolds, projectile_type)
            k = 0.5 * rho * cd * area
            drag_x = -k * speed_rel * vx_rel
            drag_y = -k * speed_rel * vy_rel
        else:  # Default to constant drag if invalid model specified
            k = 0.5 * rho * 0.47 * area
            drag_x = -k * speed_rel * vx_rel
            drag_y = -k * speed_rel * vy_rel
        
        # Magnus effect for spinning projectiles
        magnus_force_x = 0
        magnus_force_y = 0
        if spin_rate != 0:
            # Simplified magnus effect
            spin_factor = 1.0e-4 * spin_rate
            magnus_force_x = spin_factor * vy_rel
            magnus_force_y = -spin_factor * vx_rel
        
        # Coriolis force (perpendicular to velocity and Earth's rotation axis)
        coriolis_force_x = coriolis_param * vy if coriolis_enabled else 0
        coriolis_force_y = 0  # Simplified - vertical component usually negligible
        
        # Sum all forces
        fx = drag_x + magnus_force_x + coriolis_force_x
        fy = drag_y + magnus_force_y + coriolis_force_y - mass * G
        
        # Update velocities (F=ma)
        vx += (fx/mass) * dt
        vy += (fy/mass) * dt
        
        # Update positions
        x += vx * dt
        y += vy * dt
        t += dt
    
    # Ensure we have at least one data point for empty trajectories
    if not times:
        times = [0]
        x_vals = [0]
        y_vals = [launch_height]
        vx_vals = [vx]
        vy_vals = [vy]
    
    return np.array(times), np.array(x_vals), np.array(y_vals), np.array(vx_vals), np.array(vy_vals)

def simulate_projectile(
    launch_speed, launch_angle, mass, diameter,
    drag_coeff=0.47, dt=0.01, max_time=100,  # Increased max_time to prevent cutoffs
    wind_speed=0.0, wind_angle=0.0, launch_height=0.0
):
    """Simulates projectile motion with air resistance and wind"""
    # Input validation
    if mass <= 0:
        mass = 0.01  # Prevent division by zero
    if diameter <= 0:
        diameter = 0.01  # Prevent division by zero
        
    theta = np.radians(launch_angle)
    wind_theta = np.radians(wind_angle)
    
    # Wind components
    wind_x = wind_speed * np.cos(wind_theta)
    wind_y = wind_speed * np.sin(wind_theta)
    
    # Initial conditions
    vx = launch_speed * np.cos(theta)
    vy = launch_speed * np.sin(theta)
    x, y, t = 0, launch_height, 0
    
    # Pre-calculate constants
    area = np.pi * (diameter/2)**2
    k = 0.5 * AIR_DENSITY * drag_coeff * area
    
    # Storage for results
    times, x_vals, y_vals = [], [], []
    vx_vals, vy_vals = [], []  # Added to track velocities
    
    while y >= 0 and t < max_time:
        times.append(t)
        x_vals.append(x)
        y_vals.append(y)
        vx_vals.append(vx)
        vy_vals.append(vy)
        
        # Relative velocity including wind
        vx_rel = vx - wind_x
        vy_rel = vy - wind_y
        speed_rel = np.sqrt(vx_rel**2 + vy_rel**2)
        
        # Safety check for zero speed
        if speed_rel < 1e-10:
            speed_rel = 1e-10  # Prevent division by zero
        
        # Drag force components
        drag_x = -k * speed_rel * vx_rel
        drag_y = -k * speed_rel * vy_rel
        
        # Update velocities (F=ma)
        vx += (drag_x/mass) * dt
        vy += (-G + drag_y/mass) * dt
        
        # Update positions
        x += vx * dt
        y += vy * dt
        t += dt
    
    # Ensure we have at least one data point for empty trajectories
    if not times:
        times = [0]
        x_vals = [0]
        y_vals = [launch_height]
        vx_vals = [vx]
        vy_vals = [vy]
    
    return np.array(times), np.array(x_vals), np.array(y_vals), np.array(vx_vals), np.array(vy_vals)

def calculate_energy_components(times, x_vals, y_vals, vx_vals, vy_vals, mass):
    """Calculate kinetic and potential energy at each point of trajectory"""
    
    # Calculate velocities at each point
    velocities = np.sqrt(vx_vals**2 + vy_vals**2)
    
    # Calculate energies
    kinetic_energy = 0.5 * mass * velocities**2
    potential_energy = mass * G * y_vals
    total_energy = kinetic_energy + potential_energy
    
    return kinetic_energy, potential_energy, total_energy

def plot_energy_analysis(times, x_vals, y_vals, vx_vals, vy_vals, mass):
    """Generate energy analysis plots for projectile motion"""
    
    # Calculate energies directly from velocity data
    KE, PE, TE = calculate_energy_components(times, x_vals, y_vals, vx_vals, vy_vals, mass)
    
    # Create the energy plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot energy components over time
    ax1.plot(times, KE, 'r-', label='Kinetic Energy')
    ax1.plot(times, PE, 'b-', label='Potential Energy')
    ax1.plot(times, TE, 'g-', label='Total Energy')
    ax1.set_title('Energy Components Over Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Energy (J)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot energy components over position
    ax2.plot(x_vals, KE, 'r-', label='Kinetic Energy')
    ax2.plot(x_vals, PE, 'b-', label='Potential Energy')
    ax2.plot(x_vals, TE, 'g-', label='Total Energy')
    ax2.set_title('Energy Components Over Distance')
    ax2.set_xlabel('Horizontal Distance (m)')
    ax2.set_ylabel('Energy (J)')
    ax2.grid(True)
    ax2.legend()
    
    return fig

def display_equations():
    """Display mathematical equations from the research paper in proper format"""
    
    st.markdown("## Key Equations from Research")
    
    st.markdown("### Projectile Motion Kinematics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(r"""
        **Horizontal Motion:**
        $$x(t) = u\cos\theta \cdot t$$
        
        **Vertical Motion:**
        $$y(t) = u\sin\theta \cdot t - \frac{1}{2}gt^2$$
        
        **Trajectory Equation:**
        $$y = x\tan\theta - \frac{gx^2}{2u^2\cos^2\theta}$$
        """)
    
    with col2:
        st.markdown(r"""
        **Time of Flight:**
        $$T = \frac{2u\sin\theta}{g}$$
        
        **Maximum Height:**
        $$H = \frac{u^2\sin^2\theta}{2g}$$
        
        **Horizontal Range:**
        $$R = \frac{u^2\sin2\theta}{g}$$
        """)
    
    st.markdown("### Energy Considerations")
    
    st.markdown(r"""
    **Initial Energy (at launch):**
    $$E_{initial} = K.E. = \frac{1}{2}mu^2$$
    
    **At any point during flight:**
    $$E = \frac{1}{2}mv^2 + mgy$$
    
    **At maximum height:**
    $$K.E. = \frac{1}{2}m(u\cos\theta)^2, \quad P.E. = mgH$$
    """)
    
    st.markdown("### Air Resistance Model")
    
    st.markdown(r"""
    **Drag Force:**
    $$\vec{F}_d = - \frac{1}{2} \rho C_d A |\vec{v}| \vec{v}$$
    
    **Net Force Equation:**
    $$m\frac{d\vec{v}}{dt} = m\vec{g} + \vec{F}_d$$
    """)
    
    # Add explanations of variables
    with st.expander("Variable Definitions"):
        st.markdown("""
        - $u$ = initial velocity (m/s)
        - $\theta$ = launch angle (radians)
        - $t$ = time (s)
        - $g$ = acceleration due to gravity (9.81 m/s²)
        - $m$ = mass (kg)
        - $\rho$ = air density (kg/m³)
        - $C_d$ = drag coefficient
        - $A$ = cross-sectional area (m²)
        - $\vec{v}$ = velocity vector (m/s)
        """)

def research_validation_tab():
    """Create a tab to compare simulation results with research paper data"""
    
    st.header("🔬 Research Validation")
    
    st.markdown("""
    This section allows you to compare the simulation results with the experimental data 
    from the research paper "*Comprehensive Research Paper: Experimental and Theoretical Analysis 
    of Projectile Motion Using a DIY Launcher*".
    """)
    
    # Load research paper data
    research_data = {
        "Vacuum": {"theoretical_angle": 45.0, "experimental_angle": None, "uncertainty": None},
        "Standard (PP)": {"theoretical_angle": 42.3, "experimental_angle": 41.8, "uncertainty": 0.7},
        "High drag": {"theoretical_angle": 38.1, "experimental_angle": 37.5, "uncertainty": 1.2}
    }
    
    # Create experiment selection
    experiment_type = st.selectbox(
        "Select Experiment Condition",
        options=["Vacuum", "Standard (PP)", "High drag"]
    )
    
    # Display selected experiment data
    selected_data = research_data[experiment_type]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Research Paper Data")
        st.markdown(f"""
        **Condition:** {experiment_type}  
        **Theoretical Optimal Angle:** {selected_data['theoretical_angle']}°  
        **Experimental Optimal Angle:** {selected_data['experimental_angle'] if selected_data['experimental_angle'] else 'N/A'}° 
        {f"± {selected_data['uncertainty']}°" if selected_data['uncertainty'] else ""}
        """)
    
    # Run simulation to find optimal angle
    with col2:
        st.subheader("Simulation Results")
        
        if st.button("Calculate Optimal Angle"):
            with st.spinner("Running simulation to find optimal angle..."):
                # Placeholder for actual simulation
                progress_bar = st.progress(0)
                results = []
                
                # Configuration based on experiment type
                if experiment_type == "Vacuum":
                    drag_coeff = 0.0
                    projectile = "sphere"
                    mass = 2.7  # g
                    diameter = 40  # mm
                elif experiment_type == "Standard (PP)":
                    drag_coeff = 0.4
                    projectile = "ping_pong"
                    mass = 2.7  # g
                    diameter = 40  # mm
                else:  # High drag
                    drag_coeff = 0.7
                    projectile = "sphere"
                    mass = 2.7  # g
                    diameter = 60  # mm
                
                # Test different angles
                angles = np.arange(30, 61, 1)
                max_distance = 0
                optimal_angle = 0
                
                for i, angle in enumerate(angles):
                    # Update progress
                    progress_bar.progress((i+1)/len(angles))
                    
                    # Run simulation
                    times, x_vals, y_vals, _, _ = simulate_projectile(
                        15.0,  # Launch speed (m/s)
                        angle,  # Launch angle
                        mass/1000,  # Convert to kg
                        diameter/1000,  # Convert to m
                        drag_coeff
                    )
                    
                    distance = x_vals[-1] if len(x_vals) > 0 else 0
                    results.append((angle, distance))
                    
                    if distance > max_distance:
                        max_distance = distance
                        optimal_angle = angle
                
                # Display results
                st.success(f"Optimal angle from simulation: {optimal_angle:.1f}°")
                st.metric(
                    "Difference from theoretical", 
                    f"{abs(optimal_angle - selected_data['theoretical_angle']):.1f}°"
                )
                
                if selected_data['experimental_angle']:
                    st.metric(
                        "Difference from experimental", 
                        f"{abs(optimal_angle - selected_data['experimental_angle']):.1f}°"
                    )
                
                # Plot results
                results_df = pd.DataFrame(results, columns=["Angle", "Distance"])
                fig, ax = plt.subplots()
                ax.plot(results_df["Angle"], results_df["Distance"])
                ax.set_xlabel("Launch Angle (degrees)")
                ax.set_ylabel("Distance (m)")
                ax.set_title("Distance vs. Launch Angle")
                ax.axvline(x=optimal_angle, color='r', linestyle='--', label=f"Optimal: {optimal_angle:.1f}°")
                if selected_data['theoretical_angle']:
                    ax.axvline(x=selected_data['theoretical_angle'], color='g', linestyle='--', 
                              label=f"Theoretical: {selected_data['theoretical_angle']}°")
                if selected_data['experimental_angle']:
                    ax.axvline(x=selected_data['experimental_angle'], color='b', linestyle='--', 
                              label=f"Experimental: {selected_data['experimental_angle']}°")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

def create_download_link(df, filename="projectile_data.csv"):
    """Generates a download link for DataFrame"""
    try:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color: #28a745; font-weight: bold; text-decoration: none;">📥 Download CSV</a>'
    except Exception as e:
        st.error(f"Error generating download link: {str(e)}")
        return ""

def main():
    # Page configuration
    st.set_page_config(
        page_title="Advanced Projectile Simulator",
        page_icon="🚀",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #2a5885;}
    .stSlider {padding: 0px 30px;}
    .stButton>button {background-color: #4a8ddc; color: white; border-radius: 5px;}
    .st-expander {border: 1px solid #dee2e6; border-radius: 5px;}
    .metric-value {font-size: 1.2rem !important;}
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🚀 Advanced Projectile Motion Simulator")
    st.markdown("""
    *A comprehensive tool for analyzing projectile trajectories with air resistance, wind effects, and multi-parameter comparisons.*
    """)
    
    # Initialize session state
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = {}
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Simulator", "Energy Analysis", "Physics Equations", "Research Validation"])
    
    with tab1:
        # ========================================
        # Input Parameters Section
        # ========================================
        with st.sidebar:
            st.header("⚙️ Simulation Parameters")
            
            # Simulation mode selection
            sim_mode = st.radio(
                "Simulation Mode",
                ["Single Shot", "Comparative Analysis"],
                index=0
            )
            
            # Common parameters
            st.subheader("Projectile Properties")
            mass = st.slider("Mass (kg)", 0.01, 10.0, 0.1, 0.01)
            diameter = st.slider("Diameter (m)", 0.01, 1.0, 0.07, 0.01)
            drag_coeff = st.select_slider(
                "Drag Coefficient", 
                options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.47, 0.5, 0.6, 0.7],  # Added 0.0 for vacuum simulation
                value=0.47
            )
            
            # Environment parameters
            st.subheader("Environment")
            wind_speed = st.slider("Wind Speed (m/s)", 0.0, 20.0, 0.0, 0.1)
            wind_angle = st.slider("Wind Direction (°)", 0, 359, 0, 1)
            launch_height = st.slider("Launch Height (m)", 0.0, 10.0, 0.85, 0.05)
            
            # Simulation-specific parameters
            simulations = []
            if sim_mode == "Single Shot":
                st.subheader("Launch Conditions")
                launch_speed = st.slider("Speed (m/s)", 1.0, 100.0, 15.0, 0.1)
                launch_angle = st.slider("Angle (°)", 0, 90, 45, 1)
            else:
                st.subheader("Comparative Parameters")
                num_comparisons = st.number_input(
                    "Number of comparisons", 1, 5, 2, 1
                )
                
                # Generate a list of distinct colors
                colors = [hsl_to_hex(i*(360//max(num_comparisons,1)), 70, 50) for i in range(num_comparisons)]
                
                for i in range(num_comparisons):
                    with st.expander(f"Simulation {i+1}"):
                        sim = {
                            "name": st.text_input(f"Label {i+1}", f"Config {i+1}"),
                            "speed": st.slider(f"Speed {i+1} (m/s)", 1.0, 100.0, 15.0+5*i, 0.1),
                            "angle": st.slider(f"Angle {i+1} (°)", 0, 90, 45-10*i if i < 5 else 5, 1),
                            "color": st.color_picker(f"Color {i+1}", colors[i])
                        }
                        simulations.append(sim)
            
            # Run button
            if st.button("▶️ Run Simulation", use_container_width=True):
                st.session_state.run_simulation = True
                # Store simulation data for use across tabs
                st.session_state.simulation_results = {
                    "mode": sim_mode,
                    "mass": mass,
                    "diameter": diameter,
                    "drag_coeff": drag_coeff,
                    "wind_speed": wind_speed,
                    "wind_angle": wind_angle,
                    "launch_height": launch_height
                }
                
                if sim_mode == "Single Shot":
                    st.session_state.simulation_results["launch_speed"] = launch_speed
                    st.session_state.simulation_results["launch_angle"] = launch_angle
                else:
                    st.session_state.simulation_results["simulations"] = simulations

        # ========================================
        # Results Visualization Section
        # ========================================
        if st.session_state.run_simulation:
            st.header("📊 Results")
            
            # Extract stored simulation parameters
            sim_data = st.session_state.simulation_results
            mass = sim_data["mass"] 
            diameter = sim_data["diameter"]
            drag_coeff = sim_data["drag_coeff"]
            wind_speed = sim_data["wind_speed"]
            wind_angle = sim_data["wind_angle"]
            launch_height = sim_data["launch_height"]
            sim_mode = sim_data["mode"]
            
            if sim_mode == "Single Shot":
                try:
                    # Single simulation results
                    launch_speed = sim_data["launch_speed"]
                    launch_angle = sim_data["launch_angle"]
                    
                    times, x_vals, y_vals, vx_vals, vy_vals = simulate_projectile(
                        launch_speed, launch_angle, mass, diameter,
                        drag_coeff, wind_speed=wind_speed, wind_angle=wind_angle,
                        launch_height=launch_height
                    )
                    
                    # Store trajectory data for energy analysis
                    st.session_state.simulation_results["trajectory"] = {
                        "times": times,
                        "x_vals": x_vals,
                        "y_vals": y_vals,
                        "vx_vals": vx_vals,
                        "vy_vals": vy_vals
                    }
                    
                    # Calculate metrics
                    max_height = max(y_vals) if len(y_vals) > 0 else launch_height
                    total_distance = x_vals[-1] if len(x_vals) > 0 else 0
                    flight_time = times[-1] if len(times) > 0 else 0
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Max Height", f"{max_height:.2f} m")
                    col2.metric("Total Distance", f"{total_distance:.2f} m")
                    col3.metric("Flight Time", f"{flight_time:.2f} s")
                    
                    # Create plots
                    fig = plt.figure(figsize=(16, 6))
                    
                    # 2D Plot
                    ax1 = fig.add_subplot(121)
                    ax1.plot(x_vals, y_vals, 'b-', linewidth=2)
                    ax1.set_title("2D Trajectory Projection")
                    ax1.set_xlabel("Horizontal Distance (m)")
                    ax1.set_ylabel("Vertical Height (m)")
                    ax1.grid(True)
                    ax1.set_ylim(0, max(max_height*1.1, launch_height*1.1, 0.1))
                    
                    # 3D Plot
                    ax2 = fig.add_subplot(122, projection='3d')
                    ax2.plot(times, x_vals, y_vals, 'r-', linewidth=2)
                    ax2.set_title("3D Trajectory (Time Evolution)")
                    ax2.set_xlabel("Time (s)")
                    ax2.set_ylabel("Distance (m)")
                    ax2.set_zlabel("Height (m)")
                    
                    st.pyplot(fig)
                    
                    # Wind visualization
                    if wind_speed > 0:
                        fig_wind, ax = plt.subplots(figsize=(6, 4))
                        ax.quiver(0, 0, 
                                 wind_speed*np.cos(np.radians(wind_angle)),
                                 wind_speed*np.sin(np.radians(wind_angle)),
                                 scale=10, color='orange')
                        ax.set_xlim(-wind_speed*2, wind_speed*2)
                        ax.set_ylim(-wind_speed*2, wind_speed*2)
                        ax.set_title(f"Wind Vector: {wind_speed} m/s at {wind_angle}°")
                        ax.grid(True)
                        st.pyplot(fig_wind)
                    
                    # Data export
                    st.subheader("📤 Data Export")
                    df = pd.DataFrame({
                        "Time (s)": times,
                        "Distance (m)": x_vals,
                        "Height (m)": y_vals,
                        "Velocity X (m/s)": vx_vals,
                        "Velocity Y (m/s)": vy_vals
                    })
                    
                    with st.expander("View Raw Data"):
                        st.dataframe(df.style.format("{:.4f}"), height=300)
                    
                    st.markdown(create_download_link(df), unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Simulation error: {str(e)}")
            
            else:
                try:
                    # Comparative analysis results
                    fig_traj = plt.figure(figsize=(12, 6))
                    ax_traj = fig_traj.add_subplot(111)
                    
                    all_results = []
                    max_height_overall = 0
                    max_distance_overall = 0
                    
                    # Run all simulations
                    for sim in sim_data["simulations"]:
                        times, x_vals, y_vals, vx_vals, vy_vals = simulate_projectile(
                            sim["speed"], sim["angle"], mass, diameter,
                            drag_coeff, wind_speed=wind_speed, wind_angle=wind_angle,
                            launch_height=launch_height
                        )
                        
                        # Track overall metrics
                        max_height = max(y_vals) if len(y_vals) > 0 else launch_height
                        max_distance = x_vals[-1] if len(x_vals) > 0 else 0
                        flight_time = times[-1] if len(times) > 0 else 0
                        
                        max_height_overall = max(max_height_overall, max_height)
                        max_distance_overall = max(max_distance_overall, max_distance)
                        
                        # Store results
                        result = {
                            "name": sim["name"],
                            "color": sim["color"],
                            "max_height": max_height,
                            "distance": max_distance,
                            "flight_time": flight_time,
                            "trajectory": {
                                "times": times,
                                "x_vals": x_vals,
                                "y_vals": y_vals,
                                "vx_vals": vx_vals,
                                "vy_vals": vy_vals
                            }
                        }
                        all_results.append(result)
                        
                        # Plot trajectory
                        ax_traj.plot(x_vals, y_vals, color=sim["color"], linewidth=2, label=f"{sim['name']}")
                    
                    # Store results for energy analysis
                    st.session_state.simulation_results["comparative_results"] = all_results
                    
                    # Finalize plot
                    ax_traj.set_title("Comparative Trajectories")
                    ax_traj.set_xlabel("Horizontal Distance (m)")
                    ax_traj.set_ylabel("Vertical Height (m)")
                    ax_traj.grid(True)
                    ax_traj.set_ylim(0, max_height_overall*1.1)
                    ax_traj.set_xlim(0, max_distance_overall*1.05)
                    ax_traj.legend()
                    st.pyplot(fig_traj)
                    
                    # Display metrics table
                    st.subheader("Comparative Metrics")
                    metrics_df = pd.DataFrame([
                        {
                            "Configuration": r["name"],
                            "Maximum Height (m)": f"{r['max_height']:.2f}",
                            "Total Distance (m)": f"{r['distance']:.2f}",
                            "Flight Time (s)": f"{r['flight_time']:.2f}"
                        } for r in all_results
                    ])
                    
                    st.table(metrics_df)
                    
                    # Data export
                    st.subheader("📤 Data Export")
                    combined_data = []
                    for r in all_results:
                        df = pd.DataFrame({
                            "Time (s)": r["trajectory"]["times"],
                            "Distance (m)": r["trajectory"]["x_vals"],
                            "Height (m)": r["trajectory"]["y_vals"],
                            "Velocity X (m/s)": r["trajectory"]["vx_vals"],
                            "Velocity Y (m/s)": r["trajectory"]["vy_vals"],
                            "Configuration": r["name"]
                        })
                        combined_data.append(df)
                    
                    combined_df = pd.concat(combined_data)
                    
                    with st.expander("View Combined Data"):
                        st.dataframe(combined_df.style.format("{:.4f}"), height=300)
                    
                    st.markdown(
                        create_download_link(combined_df, "comparative_projectile_data.csv"),
                        unsafe_allow_html=True
                    )
                
                except Exception as e:
                    st.error(f"Comparative simulation error: {str(e)}")
    
    with tab2:
        st.header("⚡ Energy Analysis")
        
        if st.session_state.run_simulation:
            sim_data = st.session_state.simulation_results
            sim_mode = sim_data["mode"]
            mass = sim_data["mass"]
            
            if sim_mode == "Single Shot" and "trajectory" in sim_data:
                # Extract trajectory data
                trajectory = sim_data["trajectory"]
                times = trajectory["times"]
                x_vals = trajectory["x_vals"]
                y_vals = trajectory["y_vals"]
                vx_vals = trajectory["vx_vals"]
                vy_vals = trajectory["vy_vals"]
                
                # Plot energy components
                energy_fig = plot_energy_analysis(times, x_vals, y_vals, vx_vals, vy_vals, mass)
                st.pyplot(energy_fig)
                
                # Calculate velocities
                velocities = np.sqrt(vx_vals**2 + vy_vals**2)
                
                # Plot velocity profile
                vel_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Velocity components over time
                ax1.plot(times, vx_vals, 'r-', label='Horizontal Velocity')
                ax1.plot(times, vy_vals, 'b-', label='Vertical Velocity')
                ax1.plot(times, velocities, 'g-', label='Total Velocity')
                ax1.set_title('Velocity Components Over Time')
                ax1.set_xlabel('Time (s)')
                ax1.set_ylabel('Velocity (m/s)')
                ax1.grid(True)
                ax1.legend()
                
                # Velocity over position
                ax2.plot(x_vals, velocities, 'g-')
                ax2.set_title('Velocity vs. Position')
                ax2.set_xlabel('Horizontal Distance (m)')
                ax2.set_ylabel('Total Velocity (m/s)')
                ax2.grid(True)
                
                st.pyplot(vel_fig)
                
            elif sim_mode == "Comparative Analysis" and "comparative_results" in sim_data:
                # Select which simulation to analyze
                results = sim_data["comparative_results"]
                selected_sim = st.selectbox(
                    "Select Configuration for Energy Analysis",
                    options=[r["name"] for r in results]
                )
                
                # Find the selected simulation data
                selected_data = next((r for r in results if r["name"] == selected_sim), None)
                
                if selected_data:
                    # Extract trajectory data
                    trajectory = selected_data["trajectory"]
                    times = trajectory["times"]
                    x_vals = trajectory["x_vals"]
                    y_vals = trajectory["y_vals"]
                    vx_vals = trajectory["vx_vals"]
                    vy_vals = trajectory["vy_vals"]
                    
                    # Plot energy analysis
                    energy_fig = plot_energy_analysis(times, x_vals, y_vals, vx_vals, vy_vals, mass)
                    st.pyplot(energy_fig)
                    
                    # Calculate velocities
                    velocities = np.sqrt(vx_vals**2 + vy_vals**2)
                    
                    # Plot velocity profile
                    vel_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    
                    # Velocity components over time
                    ax1.plot(times, vx_vals, 'r-', label='Horizontal Velocity')
                    ax1.plot(times, vy_vals, 'b-', label='Vertical Velocity')
                    ax1.plot(times, velocities, 'g-', label='Total Velocity')
                    ax1.set_title('Velocity Components Over Time')
                    ax1.set_xlabel('Time (s)')
                    ax1.set_ylabel('Velocity (m/s)')
                    ax1.grid(True)
                    ax1.legend()
                    
                    # Velocity over position
                    ax2.plot(x_vals, velocities, 'g-')
                    ax2.set_title('Velocity vs. Position')
                    ax2.set_xlabel('Horizontal Distance (m)')
                    ax2.set_ylabel('Total Velocity (m/s)')
                    ax2.grid(True)
                    
                    st.pyplot(vel_fig)
                    
            else:
                st.info("Run a simulation first to view energy analysis.")
        else:
            st.info("Run a simulation first to view energy analysis.")
    
    with tab3:
        display_equations()
    
    with tab4:
        research_validation_tab()

if __name__ == "__main__":
    main()
