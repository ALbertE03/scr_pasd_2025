import streamlit as st
import pandas as pd
import plotly.graph_objects as go


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

def render_cluster_status_tab(cluster_status,system_metrics,api_client):
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
