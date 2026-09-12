/**
 * UgvModel.tsx -- a compact, recognizable unmanned ground vehicle built
 * from Three.js primitives (engineering-portal redesign, Section 1: no
 * GLB/GLTF asset exists in this repo, so a low-poly primitive build is
 * the right call here). Prioritized in the brief's own order --
 * SILHOUETTE > RECOGNIZABILITY > SHADING > DETAIL -- so this is
 * deliberately a simple two-tier chassis (a wide lower skirt + a
 * narrower upper deck, the classic small-vehicle-from-above silhouette)
 * with four wheels, a raised sensor mast, and a single small heading
 * indicator, not a detailed model.
 *
 * Faces +X in its own local space (this codebase's own established
 * "x is forward" convention -- see RealPointCloud.tsx's own comment on
 * the sensor frame); the parent group applies position and yaw (see
 * Scene.tsx's VehicleMarker / RealScene.tsx's RealVehicleMarker), so
 * this component itself takes no props.
 */

import { VEHICLE_COLOR, PATH_COLOR } from "../lib/theme"

const WHEEL_POSITIONS: [number, number][] = [
  [-0.16, 0.16],
  [0.16, 0.16],
  [-0.16, -0.16],
  [0.16, -0.16],
]

export function UgvModel() {
  return (
    <group castShadow>
      {/* Contact shadow -- a soft dark blob grounding the vehicle on the
          terrain regardless of shadow-map resolution/framing. */}
      <mesh position={[0, 0.005, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[0.32, 20]} />
        <meshBasicMaterial color="#000000" transparent opacity={0.28} depthWrite={false} />
      </mesh>

      {/* Lower chassis skirt -- the wide, low base that reads as
          "vehicle body" in silhouette from above. */}
      <mesh position={[0, 0.075, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.42, 0.09, 0.3]} />
        <meshStandardMaterial color={VEHICLE_COLOR.body} roughness={0.75} metalness={0.05} />
      </mesh>

      {/* Upper deck -- a narrower block stepped in from the skirt,
          faking a beveled chassis without needing rounded geometry. */}
      <mesh position={[-0.02, 0.145, 0]} castShadow receiveShadow>
        <boxGeometry args={[0.3, 0.05, 0.22]} />
        <meshStandardMaterial color={VEHICLE_COLOR.dark} roughness={0.7} metalness={0.05} />
      </mesh>

      {/* Raised sensor/LiDAR mast, slightly forward of centre. */}
      <mesh position={[0.06, 0.21, 0]} castShadow>
        <cylinderGeometry args={[0.035, 0.04, 0.07, 10]} />
        <meshStandardMaterial color={VEHICLE_COLOR.dark} roughness={0.55} metalness={0.1} />
      </mesh>
      <mesh position={[0.06, 0.25, 0]} castShadow>
        <boxGeometry args={[0.05, 0.02, 0.05]} />
        <meshStandardMaterial color={VEHICLE_COLOR.highlight} roughness={0.4} metalness={0.15} />
      </mesh>

      {/* Four wheels -- axis rotated to run across the vehicle's width. */}
      {WHEEL_POSITIONS.map(([x, z]) => (
        <mesh key={`${x}-${z}`} position={[x, 0.05, z]} rotation={[0, 0, Math.PI / 2]} castShadow>
          <cylinderGeometry args={[0.05, 0.05, 0.05, 14]} />
          <meshStandardMaterial color={VEHICLE_COLOR.dark} roughness={0.85} metalness={0} />
        </mesh>
      ))}

      {/* Single restrained heading indicator -- the ONLY place on the
          vehicle body that uses the navigation accent colour. */}
      <mesh position={[0.21, 0.1, 0]}>
        <sphereGeometry args={[0.018, 10, 10]} />
        <meshStandardMaterial color={PATH_COLOR} emissive={PATH_COLOR} emissiveIntensity={0.6} roughness={0.4} />
      </mesh>
    </group>
  )
}
