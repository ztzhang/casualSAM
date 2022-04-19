video3d_template = """
<!DOCTYPE html>
<html>

<head>
	<meta http-equiv="Content-Type" content="text/html; charset=windows-1252">
	<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
	<meta http-equiv="Pragma" content="no-cache">
	<meta http-equiv="Expires" content="0">
	<title>3D video</title>
	<script type="text/javascript" src="https://ztzhang.info/assets/three_js/jquery.min.js"></script>
	<script src="https://ztzhang.info/assets/three_js/three.js"></script>
	<script src="https://ztzhang.info/assets/three_js/PLYLoader.js"></script>
	<script src="https://ztzhang.info/assets/three_js/OrbitControls.js"></script>
	<script>var obj_files = [{ply_string}];</script>
	<style>
		#c {{
			position: fixed;
			left: 0px;
			width: 100%;
			height: 100%;
			z-index: -1
		}}
	</style>
	<link rel="stylesheet" href="https://ztzhang.info/assets/three_js/style.css">
</head>

<body>
	<center>
		<canvas id="c" width="1562" height="1174"></canvas>
		<div style="z-index: 1">
			<table vertical-align="top" height="800px">
				<tbody>
					<tr vertical-align="top">
						<td class="scene" ,="" height="800px" align="center">
							<div style="min-height: 800px; width: 800px">&nbsp;</div>
						</td>
					</tr>
				</tbody>
			</table>
		</div>
		<script src="https://ztzhang.info/assets/three_js/ply_viewer.js"></script>
	</center>


</body>

</html>
"""
