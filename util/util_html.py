
# %%
from os.path import join, dirname, exists, basename
from glob import glob
from os import makedirs

from matplotlib.pyplot import table


class Webpage():
    WEB_TEMPLATE = """
    <html>
    <head>
    <script type="text/javascript" language="javascript" src="https://code.jquery.com/jquery-3.3.1.js"></script>

    <script type="text/javascript" language="javascript"
        src="https://cdn.datatables.net/1.10.20/js/jquery.dataTables.min.js"></script>
    <script type="text/javascript" language="javascript"
        src="https://cdn.datatables.net/buttons/1.6.1/js/dataTables.buttons.min.js"></script>
    <script type="text/javascript" language="javascript"
        src="https://cdn.datatables.net/buttons/1.6.1/js/buttons.colVis.min.js"></script>
    <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/lozad/dist/lozad.min.js"></script>
    <script>
        function parseHTML(html) {{
            var t = document.createElement('template');
            t.innerHTML = html;
            return t.content;
        }}
    </script>
    
    {custom_js_content}
    
    <link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.20/css/jquery.dataTables.min.css">
    </link>
        <style>
        .plotly-graph-div{{
        margin:auto;        
    }}
    table {{
    background-color: black;
    color: white; 
    }}
    table.dataTable.compact tbody th, table.dataTable.compact tbody td {{
    padding: 4px;
    background-color: black;
    }}
    .dataTables_wrapper .dataTables_length, .dataTables_wrapper .dataTables_filter, .dataTables_wrapper .dataTables_info, .dataTables_wrapper .dataTables_processing, .dataTables_wrapper .dataTables_paginate {{
    color: #fff;
    }}

    table.dataTable.cell-border tbody th, table.dataTable.cell-border tbody td {{
    border-top: 1px solid #ddd;
    border-right: 1px solid #ddd;
    border-bottom: 1px solid white;
    }}

    table.dataTable thead th, table.dataTable thead td {{
    border-bottom: 1px solid #fff;
    }}

    

    table.dataTable thead th, table.dataTable thead td {{
    padding: 10px 18px;
    border-bottom: 1px solid #fff;
}}

    .dataTables_wrapper .dataTables_paginate .paginate_button {{
        box-sizing: border-box;
        display: inline-block;
        min-width: 1.5em;
        padding: 0.5em 1em;
        margin-left: 2px;
        text-align: center;
        text-decoration: none !important;
        cursor: pointer;
        *cursor: hand;
        color: #fff !important;
        border: 1px solid transparent;
        border-radius: 2px;
    }}
    </style>
    
    <script>
    var global_table = {{}};
        const observer = lozad(); // lazy loads elements with default selector as '.lozad'
        observer.observe();
        $(document).ready(function () {{
            global_table.table = $('#myTable').dataTable({{
                dom: 'Blfrtip',
                autoWidth: false,
                buttons: [
                    'columnsToggle'
                ],
                "lengthMenu": [[5, 10, 15, 20, -1], [5, 10, 15, 20, "All"]],
                "columnDefs": [
                    {{"targets": "_all",
                    "className": "dt-center",
                    {custom_col_defs},
                    }}
                ],
                {custom_data_table_settings}
            }});
        }});
    </script>

    </head>

    <body bgcolor='black'>
        {body_content}
    </body>
    {custom_js_content_bottom}
    </html>
"""
    image_tag_template = "<td><img src=\"{image_path}\" style=\"max-width:100%;height:auto;\"></td>"
    table_template = """
        <table id="myTable" class="cell-border compact stripe">
            <thead>
                <tr>
                    {table_header}
                </tr>
            </thead>
            <tbody>

                    {table_body}
            </tbody>
        </table>
    """

    def __init__(self, notable=False):
        self.content = self.WEB_TEMPLATE
        self.table_content = self.table_template
        self.video_content = ''
        if not notable:
            self.devider = f'<hr><div style="text-align:center; font-size:20px;color:ffffff;">data table</div><br>'
        else:
            self.devider = ''
        self.custom_js_content = ''
        self.custom_html_content = ''
        self.custom_js_content_bottom = ''
        self.custom_data_table_settings = ''
        self.custom_col_defs = ''

    def add_image_table_from_dict(self, dict_paths):
        keys = list(dict_paths.keys())
        header = ''
        for k in keys:
            if 'string' not in k:
                header += f"<th>{k}</th>\n"
        content = ""
        file_lists = {}
        l = 0
        for k in keys:
            if l == 0:
                l = len(dict_paths[k])
            else:
                assert l == len(dict_paths[k])
        for i in range(l):
            content += "<tr>\n"
            for k in keys:
                if 'string' not in k:
                    if 'link' not in k:
                        link = dict_paths[k][i]
                        content += f"<td><img src=\"{link}\" style=\"max-width:100%;height:auto;\"></td>\n"
                    else:

                        if k+'_string' in dict_paths:
                            link_text = dict_paths[k+'_string'][i]
                        else:
                            link_text = k
                        link_addr = dict_paths[k][i]
                        content += f"<td><a  href=\"{link_addr}\" style=\"color:white;\">{link_text}</a></td>\n"
            content += "</tr>\n"
        self.table_content = self.table_content.format(
            table_header=header, table_body=content)

    def add_video_table_from_dict(self, dict_paths):
        keys = list(dict_paths.keys())
        header = ''
        for k in keys:
            if 'string' not in k:
                header += f"<th>{k}</th>\n"
        content = ""
        file_lists = {}
        l = 0
        for k in keys:
            if l == 0:
                l = len(dict_paths[k])
            else:
                assert l == len(dict_paths[k])
        for i in range(l):
            content += "<tr>\n"
            for k in keys:
                if 'name' not in k:
                    if 'link' not in k:
                        link = dict_paths[k][i]
                        #content += f"<td><img src=\"{link}\" style=\"max-width:100%;height:auto;\"></td>\n"
                        # content += f'<td><video class="lozad" preload="metadata" width="100%" controls loop laysinline muted > <source src="{link}" type="video/mp4"> </video></td>'
                        content += f'<td><video id="{k}_{i:04d}" preload="metadata" width="100%" controls loop laysinline muted ></video></td>'
                else:
                    name = dict_paths[k][i]
                    content += f"<td>{name}</a></td>\n"
            content += "</tr>\n"
        self.table_content = self.table_content.format(
            table_header=header, table_body=content)

    def add_image_table_from_folder(self, path, img_prefixes, keys=None, rel_path='./'):
        if keys is None:
            keys = img_prefixes
        header = ''
        for k in keys:
            header += f"<th>{k}</th>\n"
        content = ""
        file_lists = {}

        for prefix in img_prefixes:
            file_lists[prefix] = sorted(glob(join(path, prefix + '*')))
            l = len(file_lists[prefix])
        for i in range(l):
            content += "<tr>\n"
            for k in file_lists.keys():
                link = join(rel_path, basename(file_lists[k][i]))
                content += f"<td><img src=\"{link}\" style=\"max-width:100%;height:auto;\"></td>\n"
            content += "</tr>\n"
        self.table_content = self.table_content.format(
            table_header=header, table_body=content)

    def add_video(self, rel_path_to_video, title=''):
        video_tag = f'<div style="text-align:center; font-size:20px;color:ffffff;">{title}<br><video width="40%" max-width="40%" height="auto" autoplay loop laysinline muted > <source src="{rel_path_to_video}" type="video/mp4"> </video><br><br></div>'

        self.video_content += video_tag

    def add_div(self, div_string):
        self.video_content += div_string

    def save(self, path):
        content = self.content.format(
            body_content=self.video_content + self.devider + self.table_content+self.custom_html_content, custom_js_content=self.custom_js_content, custom_js_content_bottom=self.custom_js_content_bottom, custom_data_table_settings=self.custom_data_table_settings, custom_col_defs=self.custom_col_defs)
        d = dirname(path)
        makedirs(d, exist_ok=True)
        with open(path, 'w') as f:
            f.write(content)
        return
