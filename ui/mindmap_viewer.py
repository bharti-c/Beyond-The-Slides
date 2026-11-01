"""Interactive Mind Map visualization component.

Renders a left-to-right (LR) 4-level hierarchical concept graph using Vis.js:
- Branches beyond Level 2 are collapsed by default; click any node to expand/collapse its subtree.
- Color-coded by origin ('slide-only', 'verbal-only', 'both') with legend.
- Hover tooltip and interactive inspector panel display plain-language summary,
  slide_refs, note_refs, and video timestamp.
- Clicking a node or its seek button jumps the synchronized video player to its timestamp.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components

from models import MindMap, MindMapNode
from ui.video_player import format_seconds, render_timestamp_seek_button


def flatten_mindmap_to_vis_data(
    root: MindMapNode,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert recursive MindMapNode tree into Vis.js nodes and edges with level metadata."""
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    color_map = {
        "slide-only": {
            "background": "#EFF6FF",
            "border": "#2563EB",
            "highlight": {"background": "#DBEAFE", "border": "#1D4ED8"},
        },
        "verbal-only": {
            "background": "#F5F3FF",
            "border": "#7C3AED",
            "highlight": {"background": "#EDE9FE", "border": "#6D28D9"},
        },
        "both": {
            "background": "#ECFDF5",
            "border": "#059669",
            "highlight": {"background": "#D1FAE5", "border": "#047857"},
        },
    }

    def traverse(node: MindMapNode, parent_id: Optional[str] = None, level: int = 0) -> None:
        origin = node.origin or node.source_type or "both"
        c = color_map.get(origin, color_map["both"])
        ts = node.timestamp if node.timestamp is not None else node.start_time
        ts_str = format_seconds(ts) if ts is not None else "00:00"
        has_children = len(node.children) > 0
        # Branches collapsed beyond level 2 by default (level 3 leaves start hidden)
        initially_hidden = level > 2

        slide_meta = f"Slides: {', '.join(map(str, node.slide_refs))}" if node.slide_refs else "Slides: None"
        note_meta = f"Notes: {', '.join(map(str, node.note_refs))}" if node.note_refs else "Notes: None"
        summary_text = node.summary or f"Concept node covering {node.label}."

        expand_suffix = " [+]" if (has_children and level >= 2) else ""
        display_label = f"{node.label}{expand_suffix}\n({ts_str})" if ts is not None else f"{node.label}{expand_suffix}"

        tooltip_text = (
            f"[{origin.upper()}] {node.label}\n"
            f"Summary: {summary_text}\n"
            f"{slide_meta} | {note_meta} | Time: {ts_str}"
        )

        nodes.append({
            "id": node.id,
            "label": display_label,
            "baseLabel": node.label,
            "timeLabel": ts_str,
            "title": tooltip_text,
            "summary": summary_text,
            "origin": origin,
            "slideRefs": node.slide_refs,
            "noteRefs": node.note_refs,
            "timestamp": float(ts) if ts is not None else 0.0,
            "parentId": parent_id,
            "level": level,
            "hasChildren": has_children,
            "collapsed": level >= 2 and has_children,
            "hidden": initially_hidden,
            "color": c,
            "shape": "box",
            "font": {
                "color": "#0F172A",
                "face": "Raleway, Inter, sans-serif",
                "size": 15 if level == 0 else (14 if level == 1 else 13),
                "bold": level <= 1,
            },
            "margin": 10,
            "borderWidth": 2 if level <= 1 else 1.5,
        })

        if parent_id is not None:
            edges.append({
                "id": f"edge_{parent_id}_{node.id}",
                "from": parent_id,
                "to": node.id,
                "hidden": initially_hidden,
                "color": {"color": "#94A3B8", "highlight": "#8A4F7D"},
                "arrows": "to",
                "smooth": {"type": "cubicBezier", "forceDirection": "horizontal", "roundness": 0.45},
            })

        for child in node.children:
            traverse(child, node.id, level + 1)

    traverse(root, None, 0)
    return nodes, edges


def render_interactive_mindmap(mindmap: MindMap) -> None:
    """Render interactive left-to-right collapsible Vis.js Mind Map and hierarchical concept explorer."""
    if not mindmap or not mindmap.root:
        st.info("Mind map is not yet generated.")
        return

    st.markdown(
        """
        <div style="display: flex; flex-wrap: wrap; gap: 1.2rem; align-items: center; margin-bottom: 0.8rem;">
            <span style="display: inline-flex; align-items: center; gap: 6px; font-size: 0.85rem; font-weight: 700; color: #1E40AF;">
                <span style="width: 12px; height: 12px; border-radius: 3px; background: #EFF6FF; border: 2px solid #2563EB; display: inline-block;"></span> Slide-Only
            </span>
            <span style="display: inline-flex; align-items: center; gap: 6px; font-size: 0.85rem; font-weight: 700; color: #6D28D9;">
                <span style="width: 12px; height: 12px; border-radius: 3px; background: #F5F3FF; border: 2px solid #7C3AED; display: inline-block;"></span> Verbal-Only (Professor Additions)
            </span>
            <span style="display: inline-flex; align-items: center; gap: 6px; font-size: 0.85rem; font-weight: 700; color: #047857;">
                <span style="width: 12px; height: 12px; border-radius: 3px; background: #ECFDF5; border: 2px solid #059669; display: inline-block;"></span> Covered in Both
            </span>
            <span style="font-size: 0.82rem; color: #231920; font-weight: 600; margin-left: auto;">
                Tip: Click any concept node with <code>[+]</code> to expand/collapse its detail leaves.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nodes, edges = flatten_mindmap_to_vis_data(mindmap.root)

    vis_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
      <style>
        body {{
          margin: 0;
          padding: 0;
          background-color: transparent;
          font-family: 'Raleway', -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        #network {{
          width: 100%;
          height: 460px;
          border: 1px solid #CBD5E1;
          border-radius: 14px 14px 0 0;
          background: #FFFFFF;
        }}
        #inspector {{
          padding: 12px 16px;
          background: #F8FAFC;
          border: 1px solid #CBD5E1;
          border-top: none;
          border-radius: 0 0 14px 14px;
          font-size: 13.5px;
          color: #1E293B;
          display: flex;
          flex-direction: column;
          gap: 4px;
        }}
        .insp-header {{
          display: flex;
          align-items: center;
          gap: 10px;
          font-weight: 700;
          font-size: 14.5px;
        }}
        .insp-badge {{
          font-size: 11px;
          font-weight: 800;
          text-transform: uppercase;
          padding: 2px 8px;
          border-radius: 999px;
          background: #E2E8F0;
          color: #0F172A;
        }}
        .insp-meta {{
          font-size: 12px;
          color: #475569;
          font-weight: 600;
        }}
      </style>
    </head>
    <body>
      <div id="network"></div>
      <div id="inspector">
        <div class="insp-header">
          <span id="insp-title">{mindmap.root.label}</span>
          <span class="insp-badge" id="insp-origin">{mindmap.root.origin}</span>
          <span class="insp-meta" id="insp-refs">Click any node in the tree to view its summary, slide/note refs, and toggle detail leaves.</span>
        </div>
        <div id="insp-summary">{mindmap.root.summary or "Interactive 4-level concept map of the lecture."}</div>
      </div>
      <script type="text/javascript">
        var rawNodes = {json.dumps(nodes)};
        var rawEdges = {json.dumps(edges)};
        var nodes = new vis.DataSet(rawNodes);
        var edges = new vis.DataSet(rawEdges);
        var container = document.getElementById('network');
        var data = {{ nodes: nodes, edges: edges }};
        var options = {{
          layout: {{
            hierarchical: {{
              enabled: true,
              direction: "LR",
              sortMethod: "directed",
              levelSeparation: 250,
              nodeSpacing: 110,
              treeSpacing: 130,
              blockShifting: true,
              edgeMinimization: true,
              parentCentralization: true
            }}
          }},
          interaction: {{
            hover: true,
            zoomView: true,
            dragView: true,
            navigationButtons: true
          }},
          physics: {{
            enabled: false
          }}
        }};
        var network = new vis.Network(container, data, options);

        function getDescendants(parentId) {{
          var result = [];
          var all = nodes.get();
          for (var i = 0; i < all.length; i++) {{
            if (all[i].parentId === parentId) {{
              result.push(all[i].id);
              result = result.concat(getDescendants(all[i].id));
            }}
          }}
          return result;
        }}

        function getDirectChildren(parentId) {{
          var result = [];
          var all = nodes.get();
          for (var i = 0; i < all.length; i++) {{
            if (all[i].parentId === parentId) {{
              result.push(all[i].id);
            }}
          }}
          return result;
        }}

        network.on("click", function (params) {{
          if (params.nodes.length === 0) return;
          var clickedId = params.nodes[0];
          var nodeObj = nodes.get(clickedId);
          if (!nodeObj) return;

          // Update inspector panel
          document.getElementById("insp-title").innerText = nodeObj.baseLabel;
          document.getElementById("insp-origin").innerText = nodeObj.origin;
          var sRefs = (nodeObj.slideRefs && nodeObj.slideRefs.length) ? nodeObj.slideRefs.join(", ") : "None";
          var nRefs = (nodeObj.noteRefs && nodeObj.noteRefs.length) ? nodeObj.noteRefs.join(", ") : "None";
          document.getElementById("insp-refs").innerText =
            "Slides: " + sRefs + " | Notes: " + nRefs + " | Video Time: " + nodeObj.timeLabel;
          document.getElementById("insp-summary").innerText = nodeObj.summary || "";

          // Toggle expand/collapse of children if node has children
          if (nodeObj.hasChildren && clickedId !== "root") {{
            var isCollapsed = nodeObj.collapsed;
            if (isCollapsed) {{
              // Expand direct children
              var directKids = getDirectChildren(clickedId);
              var nodeUpdates = [];
              for (var k = 0; k < directKids.length; k++) {{
                nodeUpdates.push({{ id: directKids[k], hidden: false }});
              }}
              var suffix = " [-]";
              nodeUpdates.push({{
                id: clickedId,
                collapsed: false,
                label: nodeObj.baseLabel + suffix + "\\n(" + nodeObj.timeLabel + ")"
              }});
              nodes.update(nodeUpdates);

              var allEdges = edges.get();
              var edgeUpdates = [];
              for (var e = 0; e < allEdges.length; e++) {{
                if (allEdges[e].from === clickedId) {{
                  edgeUpdates.push({{ id: allEdges[e].id, hidden: false }});
                }}
              }}
              edges.update(edgeUpdates);
            }} else {{
              // Collapse all descendants
              var descIds = getDescendants(clickedId);
              var descSet = {{}};
              var nodeUpdates = [];
              for (var d = 0; d < descIds.length; d++) {{
                descSet[descIds[d]] = true;
                nodeUpdates.push({{ id: descIds[d], hidden: true }});
              }}
              nodeUpdates.push({{
                id: clickedId,
                collapsed: true,
                label: nodeObj.baseLabel + " [+]" + "\\n(" + nodeObj.timeLabel + ")"
              }});
              nodes.update(nodeUpdates);

              var allEdges = edges.get();
              var edgeUpdates = [];
              for (var e = 0; e < allEdges.length; e++) {{
                if (descSet[allEdges[e].to]) {{
                  edgeUpdates.push({{ id: allEdges[e].id, hidden: true }});
                }}
              }}
              edges.update(edgeUpdates);
            }}
          }}
        }});
      </script>
    </body>
    </html>
    """

    components.html(vis_html, height=565)

    # Hierarchical Tree Explorer with Summaries, Metadata Refs & Video Seek Buttons
    st.markdown("### Concept Hierarchy & Video Jump Points")
    st.caption(
        "Expand any major topic below to inspect subtopic summaries, detail leaves, slide/note references, and jump the video player."
    )

    for topic in mindmap.root.children:
        topic_ts = topic.timestamp if topic.timestamp is not None else topic.start_time
        topic_header = f"{topic.label}  —  [{topic.origin}]"
        with st.expander(topic_header, expanded=False):
            if topic.summary:
                st.markdown(f"*{topic.summary}*")
            if topic_ts is not None:
                render_timestamp_seek_button(
                    topic_ts,
                    label=f"Seek Video to {topic.label} ({format_seconds(topic_ts)})",
                    key=f"mm_seek_topic_{topic.id}",
                )

            for sub in topic.children:
                sub_ts = sub.timestamp if sub.timestamp is not None else sub.start_time
                s_refs = f"Slides: {', '.join(map(str, sub.slide_refs))}" if sub.slide_refs else ""
                n_refs = f"Notes: {', '.join(map(str, sub.note_refs))}" if sub.note_refs else ""
                meta_parts = [p for p in (s_refs, n_refs) if p]
                meta_str = f" ({' | '.join(meta_parts)})" if meta_parts else ""

                col_sub, col_btn = st.columns([4.2, 1.3])
                with col_sub:
                    st.markdown(
                        f"**{sub.label}** &nbsp; `{sub.origin}`{meta_str}<br>"
                        f"<span style='font-size: 0.9rem;'>{sub.summary}</span>",
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if sub_ts is not None:
                        render_timestamp_seek_button(
                            sub_ts,
                            label=format_seconds(sub_ts),
                            key=f"mm_seek_sub_{sub.id}",
                        )

                for leaf in sub.children:
                    leaf_ts = leaf.timestamp if leaf.timestamp is not None else leaf.start_time
                    col_l, col_lb = st.columns([4.2, 1.3])
                    with col_l:
                        st.markdown(
                            f"&nbsp;&nbsp;&nbsp;&nbsp;• **{leaf.label}** (`{leaf.origin}`): {leaf.summary}",
                            unsafe_allow_html=True,
                        )
                    with col_lb:
                        if leaf_ts is not None:
                            render_timestamp_seek_button(
                                leaf_ts,
                                label=format_seconds(leaf_ts),
                                key=f"mm_seek_leaf_{leaf.id}",
                            )
