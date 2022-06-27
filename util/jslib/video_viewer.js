let container;
var renderer;
var index = 0;
var global_index = 0;
var prev_index = 0;
const timeIterval = 100;
let camera;
var scene = new THREE.Scene();
const control = [];
init();
const animate_id = setInterval(animate, timeIterval);

function init() {
    
    renderer = new THREE.WebGLRenderer( { antialias: true } );
    renderer.setPixelRatio( window.devicePixelRatio );
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild(renderer.domElement);
    scene.background = new THREE.Color( 0x000000);
    // animate();
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.z = 0.0;
    camera.position.y = 1.0;
    camera.position.x = 0.0;
    camera.up = new THREE.Vector3(0, 0, 1);
    scene.userData.camera = camera;
    let spotLight;
    var pos = [40, 0 - 40];
    var h = [50, -90];
    for (let j = 0; j < 3; j++) {
        for (let k = 0; k < 2; k++) {
            for (let l = 0; l < 3; l++) {
                spotLight = new THREE.SpotLight(0xffffff, 5);
                spotLight.position.set(pos[j], h[k], pos[l]);
                spotLight.angle = Math.PI / 4;
                spotLight.penumbra = 0.05;
                spotLight.decay = 2;
                spotLight.distance = 200;
                spotLight.castShadow = true;
                spotLight.shadow.mapSize.width = 1024;
                spotLight.shadow.mapSize.height = 1024;
                spotLight.shadow.camera.near = 10;
                spotLight.shadow.camera.far = 200;
                scene.add(spotLight);
            }
        }
    }
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    // controls.target.set( 0, 0, 0 );
    // controls.minDistance = 0.1;
    // controls.maxDistance = 1000;
    controls.maxPolarAngle = Math.PI * 2;
    controls.maxPolarAngle = Math.PI * 2;
    controls.minAzimuthAngle = - Infinity; // radians
    // controls.maxAzimuthAngle = Infinity; // radians

    controls.addEventListener('change', render);
    console.log(controls);
    control.push(controls);
    loadAllFiles();
    onWindowResize();
    
    // render();
}




function onWindowResize() {

    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();

    renderer.setSize(window.innerWidth, window.innerHeight);

}

function loadAllFiles() {
    for (const point_geometry of point_geometry_list) {
        point_geometry.visible = false;
        scene.add(point_geometry);
    }
    if (camera_geometry_list.length > 0) {
        for(const camera_geometry of camera_geometry_list){
            camera_geometry.visible = true;
            scene.add(camera_geometry);
        }
    }
    return;
}

function animate() {
    control[0].update();
    const idx = global_index % point_geometry_list.length
    point_geometry_list[prev_index].visible = false;
    point_geometry_list[idx].visible = true;
    if (camera_geometry_list.length > 0) {
        camera_geometry_list[prev_index].material.color = new THREE.Color( 0x5c5c5c );
        camera_geometry_list[prev_index].material.needsUpdate = true;
        camera_geometry_list[idx].material.color = new THREE.Color( 0xff0000 ); //.setRGB(1, 0, 0);
        camera_geometry_list[idx].material.needsUpdate = true;
        
    }
    prev_index = idx;
    global_index += 1;
    global_index %= point_geometry_list.length;
    render();
    // requestAnimationFrame(animate);
}


function render() {
    renderer.render(scene, camera);
    }
