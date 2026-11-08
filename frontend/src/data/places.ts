import type { LngLat } from './grid'

export interface City {
  id: string
  name: string
  at: LngLat
  /** Higher shows first when badges collide. */
  priority: number
}

// The 17 LGUs of the National Capital Region; points are approximate centers.
export const CITIES: City[] = [
  { id: 'qc', name: 'Quezon City', at: [121.0437, 14.676], priority: 17 },
  { id: 'manila', name: 'Manila', at: [120.9842, 14.5995], priority: 16 },
  { id: 'caloocan', name: 'Caloocan', at: [120.972, 14.6507], priority: 15 },
  { id: 'makati', name: 'Makati', at: [121.0244, 14.5547], priority: 14 },
  { id: 'pasig', name: 'Pasig', at: [121.0851, 14.5764], priority: 13 },
  { id: 'taguig', name: 'Taguig', at: [121.0509, 14.5176], priority: 12 },
  { id: 'paranaque', name: 'Parañaque', at: [121.0198, 14.4793], priority: 11 },
  { id: 'valenzuela', name: 'Valenzuela', at: [120.983, 14.7011], priority: 10 },
  { id: 'laspinas', name: 'Las Piñas', at: [120.9939, 14.4445], priority: 9 },
  { id: 'muntinlupa', name: 'Muntinlupa', at: [121.0415, 14.4081], priority: 8 },
  { id: 'marikina', name: 'Marikina', at: [121.1029, 14.6507], priority: 7 },
  { id: 'pasay', name: 'Pasay', at: [121.0014, 14.5378], priority: 6 },
  { id: 'mandaluyong', name: 'Mandaluyong', at: [121.0359, 14.5794], priority: 5 },
  { id: 'malabon', name: 'Malabon', at: [120.9567, 14.6625], priority: 4 },
  { id: 'sanjuan', name: 'San Juan', at: [121.0355, 14.6019], priority: 3 },
  { id: 'navotas', name: 'Navotas', at: [120.947, 14.657], priority: 2 },
  { id: 'pateros', name: 'Pateros', at: [121.0685, 14.5446], priority: 1 },
]

// Approximate corridors used only to shape the synthetic demo field.
export const CORRIDORS: { weight: number; line: LngLat[] }[] = [
  { // EDSA
    weight: 1,
    line: [[120.998, 14.535], [121.028, 14.549], [121.034, 14.556], [121.045, 14.566], [121.057, 14.588],
      [121.058, 14.601], [121.051, 14.62], [121.038, 14.64], [121.03, 14.657], [121.004, 14.658], [120.984, 14.657]],
  },
  { // C5
    weight: 0.8,
    line: [[121.04, 14.49], [121.05, 14.53], [121.055, 14.55], [121.075, 14.58], [121.08, 14.62], [121.075, 14.65], [121.06, 14.69]],
  },
  { // Roxas, Taft, Rizal Avenue
    weight: 0.85,
    line: [[120.995, 14.52], [120.99, 14.56], [120.98, 14.6], [120.983, 14.63], [120.985, 14.66]],
  },
  { // Commonwealth
    weight: 0.7,
    line: [[121.05, 14.64], [121.08, 14.69], [121.09, 14.72]],
  },
  { // SLEX
    weight: 0.75,
    line: [[121.02, 14.55], [121.04, 14.5], [121.04, 14.42], [121.07, 14.34]],
  },
  { // NLEX
    weight: 0.7,
    line: [[120.99, 14.66], [120.97, 14.72], [120.96, 14.78]],
  },
  { // Aurora, Marcos Highway
    weight: 0.55,
    line: [[121.02, 14.617], [121.06, 14.625], [121.1, 14.625], [121.15, 14.625]],
  },
]

export const HOTSPOTS: { at: LngLat; amp: number; radius: number }[] = [
  { at: [120.965, 14.6], amp: 13, radius: 1800 }, // port area and Tondo
  { at: [120.97, 14.7], amp: 8, radius: 2200 }, // Valenzuela industrial belt
  { at: [120.975, 14.655], amp: 6, radius: 1600 }, // Monumento
  { at: [121.02, 14.555], amp: 4, radius: 1400 }, // Makati CBD
  { at: [121.058, 14.585], amp: 5, radius: 1300 }, // Ortigas
]

export const GREEN: { at: LngLat; amp: number; radius: number }[] = [
  { at: [121.085, 14.735], amp: -9, radius: 3200 }, // La Mesa watershed
  { at: [121.065, 14.655], amp: -3, radius: 1200 }, // UP Diliman
  { at: [121.03, 14.56], amp: -2, radius: 700 }, // Forbes Park
]

// Rough shorelines. Cells inside are water and carry no estimate.
export const WATER: LngLat[][] = [
  [ // Manila Bay
    [120.9, 14.8], [120.945, 14.8], [120.935, 14.7], [120.945, 14.65], [120.955, 14.61], [120.965, 14.585],
    [120.982, 14.56], [120.98, 14.535], [120.985, 14.5], [120.975, 14.47], [120.95, 14.45], [120.91, 14.44], [120.9, 14.44],
  ],
  [ // Laguna de Bay
    [121.075, 14.53], [121.1, 14.535], [121.13, 14.545], [121.17, 14.55], [121.17, 14.3], [121.05, 14.3],
    [121.045, 14.4], [121.05, 14.45], [121.06, 14.5],
  ],
]
