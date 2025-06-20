import streamlit as st
import ray
import os
import psutil
import pandas as pd
import plotly.graph_objects as go
import time
import subprocess

@st.cache_data(ttl=30)
def get_cluster_status():
    try:
        if not ray.is_initialized():
            head_address = os.getenv('RAY_HEAD_SERVICE_HOST', 'ray-head')
            ray.init(address=f"ray://{head_address}:10001", ignore_reinit_error=True)
        
        cluster_resources = ray.cluster_resources()
        nodes = ray.nodes()
        
        alive_nodes = []
        dead_nodes = []
        for node in nodes:
            if node.get('Alive', False):
                alive_nodes.append(node)
            else:
                dead_nodes.append(node)
        
        resource_usage = {}
        for node in alive_nodes:
            node_id = node.get('NodeID', '')
            resources = node.get('Resources', {})
            available_resources = node.get('AvailableResources', {})
            
            if node_id:
                resource_usage[node_id] = {
                    'cpu_total': resources.get('CPU', 0),
                    'cpu_available': available_resources.get('CPU', 0),
                    'memory_total': resources.get('memory', 0),
                    'memory_available': available_resources.get('memory', 0),
                    'gpu_total': resources.get('GPU', 0) if 'GPU' in resources else 0,
                    'gpu_available': available_resources.get('GPU', 0) if 'GPU' in available_resources else 0
                }
        
        return {
            "connected": True,
            "resources": cluster_resources,
            "nodes": nodes,
            "alive_nodes": alive_nodes,
            "dead_nodes": dead_nodes,
            "total_cpus": cluster_resources.get('CPU', 0),
            "total_memory": cluster_resources.get('memory', 0),
            "total_gpus": cluster_resources.get('GPU', 0),
            "node_count": len(nodes),
            "alive_node_count": len(alive_nodes),
            "dead_node_count": len(dead_nodes),
            "resource_usage": resource_usage,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
            "resources": {},
            "nodes": [],
            "alive_nodes": [],
            "dead_nodes": [],
            "total_cpus": 0,
            "total_memory": 0,
            "total_gpus": 0,
            "node_count": 0,
            "alive_node_count": 0,
            "dead_node_count": 0,
            "resource_usage": {},
            "timestamp": time.time()
        }

@st.cache_data(ttl=10)
def get_system_metrics():
    """Obtiene métricas del sistema"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available": memory.available / (1024**3),  # GB
            "memory_total": memory.total / (1024**3),  # GB
            "disk_percent": disk.percent,
            "disk_free": disk.free / (1024**3),  # GB
            "disk_total": disk.total / (1024**3)  # GB
        }
    except Exception as e:
        st.error(f"Error obteniendo métricas del sistema: {e}")
        return {}

def plot_cluster_metrics(cluster_status):
    """Crea gráficos de métricas del cluster"""
    if not cluster_status['connected']:
        st.error("Cluster no conectado")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_cpus = cluster_status['total_cpus']
        
        max_cpu = max(total_cpus, 4)  

        
        fig_cpu = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_cpus,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CPUs Totales", 'font': {'size': 24, 'color': '#4361ee'}},
            gauge={
                'axis': {'range': [0, max_cpu], 'tickwidth': 1},
                'bar': {'color': "#4361ee"},
                'steps': [
                    {'range': [0, max_cpu], 'color': "rgba(67, 97, 238, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75, 
                    'value': total_cpus-0.75  
                }
            }        ))
        fig_cpu.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_cpu, use_container_width=True, key="cluster_cpu_gauge")
    
    with col2:
        memory_gb = cluster_status['total_memory'] / (1024**3) if cluster_status['total_memory'] else 0
        
        max_mem = max(memory_gb, 4) 
        
        fig_mem = go.Figure(go.Indicator(
            mode="gauge+number",
            value=memory_gb,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Memoria Total (GB)", 'font': {'size': 24, 'color': '#38b000'}},
            number={'suffix': " GB", 'valueformat': '.1f'},
            gauge={
                'axis': {'range': [0, max_mem], 'tickwidth': 1},
                'bar': {'color': "#38b000"},
                'steps': [
                    {'range': [0, max_mem], 'color': "rgba(56, 176, 0, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': memory_gb-0.75 
                }
            }
        ))
        fig_mem.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_mem, use_container_width=True, key="cluster_memory_gauge")
    
    with col3:
        total_nodes = cluster_status['node_count']
        alive_count = cluster_status['alive_node_count']
        dead_count = cluster_status['dead_node_count']
        
        if total_nodes > 0:
            alive_percent = (alive_count / total_nodes) * 100
            dead_percent = (dead_count / total_nodes) * 100
        else:
            alive_percent = 0
            dead_percent = 0
            
        node_text = [
            f"{alive_count} ({alive_percent:.1f}%)",
            f"{dead_count} ({dead_percent:.1f}%)"
        ]
        
        fig_nodes = go.Figure()

        fig_nodes.add_trace(go.Bar(
            x=['Vivos', 'Muertos'],
            y=[alive_count, dead_count],
            marker_color=['#38b000', '#ff0054'],
            text=node_text,
            textposition='auto',
            hoverinfo='y+text',
            width=[0.4, 0.4],
            hovertemplate='%{x}: %{y} nodos<br>%{text}<extra></extra>'
        ))
        
        if dead_count > 0:
            fig_nodes.add_shape(
                type="line",
                x0=0.7,  
                y0=dead_count,
                x1=1.3, 
                y1=dead_count,
                line=dict(
                    color="Red",
                    width=3,
                    dash="solid",
                ),
                name="Nodos muertos"
            )
            
            # Añadir anotación explicativa
            fig_nodes.add_annotation(
                x=1.3,
                y=dead_count,
                text=f"Nodos muertos: {dead_count}",
                showarrow=True,
                arrowhead=1,
                ax=40,
                ay=-20,
                font=dict(
                    color="red",
                    size=10
                )
            )
        fig_nodes.update_layout(
            title={
                'text': "Estado de Nodos",
                'y':0.9,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 24, 'color': '#333'}
            },
            yaxis_title="Cantidad",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_nodes, use_container_width=True, key="cluster_nodes_status")

def render_cluster_status_tab(cluster_status,system_metrics):
    """Renderiza la pestaña de estado detallado del cluster"""
    st.header("Estado Detallado del Cluster")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Nodos Vivos",
            value=cluster_status['alive_node_count'],
            delta="Online" if cluster_status['alive_node_count'] > 0 else "Ninguno"
        )
    
    with col2:
        dead_count = cluster_status['dead_node_count']
        delta_color = "inverse" if dead_count > 0 else "normal"
        st.metric(
            label="Nodos Muertos",
            value=dead_count,
            delta="⚠️ Atención" if dead_count > 0 else "✅ OK",
            delta_color=delta_color
        )
    
    with col3:
        st.metric(
            label="CPUs Totales",
            value=f"{cluster_status['total_cpus']:.0f}",
            delta=f"{system_metrics.get('cpu_percent', 0):.1f}% uso" if system_metrics else "N/A"
        )
    
    with col4:
        memory_gb = cluster_status['total_memory'] / (1024**3) if cluster_status['total_memory'] else 0
        st.metric(
            label="Memoria Total",
            value=f"{memory_gb:.1f} GB"
        )
    
    with col5:
        st.metric(
            label="GPUs",
            value=cluster_status['total_gpus'],
            delta="Disponibles" if cluster_status['total_gpus'] > 0 else "No disponibles"
        )
    
    if cluster_status['connected']:
        if cluster_status['dead_node_count'] > 0:
            st.error(
                f"⚠️ **ATENCIÓN**: {cluster_status['dead_node_count']} nodo(s) están muertos. "
                f"Solo {cluster_status['alive_node_count']} de {cluster_status['node_count']} nodos están operativos."
            )
        else:
            st.success(f"✅ Todos los {cluster_status['alive_node_count']} nodos están operativos")
        
        plot_cluster_metrics(cluster_status)
    if cluster_status['connected']:        # Sección para gestión de nodos (añadir/eliminar)
        st.subheader("� Gestión de Nodos del Cluster")
        
        tab1, tab2 = st.tabs(["📌 Añadir Nodo", "🗑️ Eliminar Nodo"])
        
        with tab1:
            
            add_cpu = st.slider(
                "Cantidad de CPUs para el nuevo nodo",
                min_value=1,
                max_value=2,
                value=2,
                step=1,
                help="Cantidad de CPUs que tendrá el nuevo nodo"
            )   
            with st.form("add_node_form"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    worker_name = st.text_input("Nombre del Nodo", 
                                               key="worker_name", 
                                               placeholder="ej: worker-gpu1",
                                               help="Nombre personalizado para identificar este nodo en el cluster")
            with col2:
                st.write("")
                st.write("")
                add_node_btn = st.form_submit_button("Añadir Nodo", 
                                                    use_container_width=True,
                                                    type="primary")
            if add_node_btn and worker_name:
                with st.spinner(f"Añadiendo nuevo nodo '{worker_name}' al cluster..."):
                    success = add_external_worker(worker_name,add_cpu)
                    
                    if success:
                        st.info("Puede tardar unos segundos en reflejarse en el dashboard del cluster.")
                        st.cache_data.clear()
                        st.rerun()
            elif not worker_name and add_node_btn:
                st.warning("Por favor ingresa un nombre para el nodo")
        
        with tab2:            
            ray_nodes = get_all_ray_nodes()
            
            if not ray_nodes:
                st.warning("No se encontraron nodos Ray para eliminar.")
                st.info("Parece que el cluster no está en ejecución o no hay contenedores Docker detectables.")
            else:
                st.success(f"Se encontraron {len(ray_nodes)} nodos en el cluster Ray.")
                
                with st.form("remove_node_form"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        node_to_remove = st.selectbox(
                            "Selecciona el Nodo a Eliminar", 
                            options=ray_nodes,
                            key="node_to_remove", 
                            help="Selecciona el nodo que deseas eliminar del cluster"
                        )
                    with col2:
                        st.write("")
                        st.write("")
                        remove_node_btn = st.form_submit_button("Eliminar Nodo", 
                                                            use_container_width=True,
                                                            type="primary")
                    
                    if remove_node_btn and node_to_remove:
                        with st.spinner(f"Eliminando nodo '{node_to_remove}' del cluster..."):
                            success = remove_ray_node(node_to_remove)
                            
                            if success:
                                st.info("El nodo ha sido eliminado. Puede tardar unos segundos en reflejarse en el dashboard.")
                                st.cache_data.clear()
                                st.rerun()
        
        st.divider()
        
        st.subheader("📊 Distribución de Recursos por Nodo")

        if cluster_status['alive_nodes']:
            node_ids = [node.get('NodeID', f'Node-{i}')[:8] + "..." for i, node in enumerate(cluster_status['alive_nodes'])]
            tab1, = st.tabs(["Recursos Totales"])
            
            with tab1:
                cpus = [node.get('Resources', {}).get('CPU', 0) for node in cluster_status['alive_nodes']]
                memory_gb = [node.get('Resources', {}).get('memory', 0) / (1024**3) for node in cluster_status['alive_nodes']]
                gpus = [node.get('Resources', {}).get('GPU', 0) for node in cluster_status['alive_nodes']]
                
                fig_node_resources = go.Figure()
                
                fig_node_resources.add_trace(go.Bar(
                    x=node_ids,
                    y=cpus,
                    name='CPUs',
                    marker_color='#4361ee'
                ))
                
                fig_node_resources.add_trace(go.Bar(
                    x=node_ids,
                    y=memory_gb,
                    name='Memoria (GB)',
                    marker_color='#38b000'
                ))
                
                fig_node_resources.add_trace(go.Bar(
                    x=node_ids,
                    y=gpus,
                    name='GPUs',
                    marker_color='#ff0054'
                ))
                
                fig_node_resources.update_layout(
                    title="Recursos Totales por Nodo",
                    xaxis_title="ID del Nodo",
                    yaxis_title="Cantidad",
                    barmode='group',
                    height=400,
                    margin=dict(l=20, r=20, t=50, b=100),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3)
                )
                
                st.plotly_chart(fig_node_resources, use_container_width=True)
            
            
        else:
            st.warning("No hay nodos activos para mostrar recursos")
        
        st.subheader("Detalle de Nodos")
        
        nodes_data = []
        for i, node in enumerate(cluster_status['nodes']):
            is_alive = node.get('Alive', False)
            node_id = node.get('NodeID', f'node-{i+1}')
            
            nodes_data.append({
                'ID': node_id,
                'Estado': "🟢 Vivo" if is_alive else "🔴 Muerto",
                'Dirección': node.get('NodeManagerAddress', 'N/A'),
                'CPU': node.get('Resources', {}).get('CPU', 0),
                'Memoria (MB)': node.get('Resources', {}).get('memory', 0) / (1024*1024) if node.get('Resources', {}).get('memory') else 0,
                'GPU': node.get('Resources', {}).get('GPU', 0),
                'Última Actualización': node.get('Timestamp', 'N/A')
            })
        if nodes_data:
            df_nodes = pd.DataFrame(nodes_data)
            st.dataframe(df_nodes, use_container_width=True)
            
            if cluster_status['dead_node_count'] > 0:
                st.warning(f"⚠️ Hay {cluster_status['dead_node_count']} nodo(s) que no responden. Esto puede afectar el rendimiento del cluster.")
        
    else:
        st.error("Cluster no conectado")
        st.warning(f"Error: {cluster_status.get('error', 'Desconocido')}")
        st.info("Por favor verifica que el cluster Ray esté en ejecución y sea accesible desde esta máquina.")


def add_external_worker(worker_name,add_cpu):
    """Añade un worker externo usando docker directamente"""
    try:
        in_docker = os.path.exists('/.dockerenv')
        
        if in_docker:
            socket_exists = os.path.exists('/var/run/docker.sock')
            if not socket_exists:
                st.error("No se puede acceder al socket de Docker desde dentro del contenedor.")
                st.info("Para añadir workers externos, ejecuta este comando desde el host o asegúrate de montar el socket de Docker.")
                return False
            command = f"""
            docker run -d --name ray_worker_{worker_name} \
            --hostname ray_worker_{worker_name} \
            --network scr_pasd_2025_ray-network \
            -e RAY_HEAD_SERVICE_HOST=ray-head \
            -e NODE_ROLE=worker \
            -e LEADER_NODE=false \
            -e FAILOVER_PRIORITY=3 \
            -e ENABLE_AUTO_FAILOVER=false \
            --shm-size=2gb \
            scr_pasd_2025-ray-head \
            bash -c "echo 'Worker externo iniciando...' && \
                    echo 'Esperando al cluster principal...' && \
                    sleep 10 && \
                    echo 'Conectando al cluster existente...' && \
                    ray start --address=ray-head:6379 --num-cpus={add_cpu} --object-manager-port=8076 \
                    --node-manager-port=8077 --min-worker-port=10002 --max-worker-port=19999 && \
                    echo 'Worker externo conectado exitosamente!' && \
                    tail -f /dev/null"
            """
        else:
            st.info("ejecute el contenedor de docker ")
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True
        )
        
        if result.returncode == 0:
            st.success(f"Worker externo 'ray_worker_{worker_name}' añadido exitosamente al cluster")
            return True
        else:
            st.error(f"Error al añadir worker externo: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"Excepción al añadir worker externo: {str(e)}")
        return False

def remove_ray_node(node_name):
    """Elimina un nodo Ray usando su nombre"""
    try:
        if node_name.startswith("ray-head") or node_name == "ray-head":
            st.info("No se puede eliminar el nodo principal (ray-head), ya que es necesario para el funcionamiento del cluster")
            return False
            
        command = f"docker stop {node_name} && docker rm {node_name}"
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True
        )
        
        if result.returncode == 0:
            st.success(f"Nodo '{node_name}' eliminado exitosamente del cluster")
            return True
        else:
            st.error(f"Error al eliminar nodo: {result.stderr}")
            return False
    except Exception as e:
        st.error(f"Excepción al eliminar nodo: {str(e)}")
        return False

def get_all_ray_nodes():
    """Obtiene la lista de todos los nodos Ray ejecutándose actualmente"""
    try:
 
        command = "docker ps --filter 'name=ray' --format '{{.Names}}'"
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            shell=True
        )
        
        if result.returncode == 0:
            node_names = result.stdout.strip().split('\n')
            node_names = [name for name in node_names if name]
            return node_names
        else:
            st.warning(f"No se pudieron listar los nodos: {result.stderr}")
            return []
    except Exception as e:
        st.warning(f"Error al obtener nodos: {str(e)}")
        return []


