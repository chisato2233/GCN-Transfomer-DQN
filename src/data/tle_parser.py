"""
TLE (Two-Line Element) Parser for LEO Satellite Data.

Parses TLE data from CelesTrak and computes satellite orbital positions.
Data sources:
- Starlink: https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle
- Iridium: https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-NEXT&FORMAT=tle
- OneWeb: https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import math


@dataclass
class SatelliteData:
    """Parsed satellite data from TLE."""
    name: str
    norad_id: int
    inclination: float      # degrees
    raan: float            # Right Ascension of Ascending Node (degrees)
    eccentricity: float
    arg_perigee: float     # degrees
    mean_anomaly: float    # degrees
    mean_motion: float     # revolutions per day
    epoch_year: int
    epoch_day: float

    # Derived properties
    altitude_km: float = 0.0
    period_minutes: float = 0.0

    def __post_init__(self):
        # Calculate altitude from mean motion
        # Using vis-viva equation: n = sqrt(GM/a^3) where a is semi-major axis
        GM = 398600.4418  # km^3/s^2 (Earth's gravitational parameter)
        EARTH_RADIUS = 6371.0  # km

        n_rad_per_sec = self.mean_motion * 2 * np.pi / 86400  # Convert to rad/s
        if n_rad_per_sec > 0:
            a = (GM / (n_rad_per_sec ** 2)) ** (1/3)  # Semi-major axis in km
            self.altitude_km = a - EARTH_RADIUS
            self.period_minutes = 1440 / self.mean_motion  # minutes


class TLEParser:
    """
    Parser for Two-Line Element (TLE) satellite data.

    TLE Format:
    Line 0: Satellite Name
    Line 1: 1 NNNNNC NNNNNAAA NNNNN.NNNNNNNN +.NNNNNNNN +NNNNN-N +NNNNN-N N NNNNN
    Line 2: 2 NNNNN NNN.NNNN NNN.NNNN NNNNNNN NNN.NNNN NNN.NNNN NN.NNNNNNNNNNNNNN
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.satellites: Dict[str, List[SatelliteData]] = {}

    def parse_tle_file(self, filepath: str, constellation_name: str = "unknown") -> List[SatelliteData]:
        """Parse a TLE file and return list of satellite data."""
        satellites = []

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # TLE comes in groups of 3 lines
        i = 0
        while i < len(lines) - 2:
            name_line = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()

            # Validate TLE format
            if not line1.startswith('1') or not line2.startswith('2'):
                i += 1
                continue

            try:
                sat = self._parse_tle_lines(name_line, line1, line2)
                if sat:
                    satellites.append(sat)
            except Exception as e:
                pass  # Skip malformed entries

            i += 3

        self.satellites[constellation_name] = satellites
        return satellites

    def _parse_tle_lines(self, name: str, line1: str, line2: str) -> Optional[SatelliteData]:
        """Parse individual TLE lines."""
        try:
            # Line 1 parsing
            norad_id = int(line1[2:7])
            epoch_year = int(line1[18:20])
            epoch_year = 2000 + epoch_year if epoch_year < 57 else 1900 + epoch_year
            epoch_day = float(line1[20:32])

            # Line 2 parsing
            inclination = float(line2[8:16])
            raan = float(line2[17:25])
            eccentricity = float('0.' + line2[26:33])
            arg_perigee = float(line2[34:42])
            mean_anomaly = float(line2[43:51])
            mean_motion = float(line2[52:63])

            return SatelliteData(
                name=name.strip(),
                norad_id=norad_id,
                inclination=inclination,
                raan=raan,
                eccentricity=eccentricity,
                arg_perigee=arg_perigee,
                mean_anomaly=mean_anomaly,
                mean_motion=mean_motion,
                epoch_year=epoch_year,
                epoch_day=epoch_day
            )
        except (ValueError, IndexError):
            return None

    def load_all_constellations(self) -> Dict[str, List[SatelliteData]]:
        """Load all available constellation TLE files."""
        tle_files = {
            'starlink': self.data_dir / 'starlink_tle.txt',
            'iridium': self.data_dir / 'iridium_tle.txt',
            'oneweb': self.data_dir / 'oneweb_tle.txt'
        }

        for name, filepath in tle_files.items():
            if filepath.exists():
                self.parse_tle_file(str(filepath), name)

        return self.satellites

    def get_satellite_positions(
        self,
        constellation: str,
        time_offset_seconds: float = 0,
        num_satellites: Optional[int] = None
    ) -> List[Dict]:
        """
        Compute satellite positions at a given time.

        Uses simplified SGP4-like propagation for position estimation.
        Returns positions in ECEF (Earth-Centered Earth-Fixed) coordinates.
        """
        if constellation not in self.satellites:
            return []

        sats = self.satellites[constellation]
        if num_satellites:
            sats = sats[:num_satellites]

        positions = []
        EARTH_RADIUS = 6371.0  # km

        for sat in sats:
            # Simplified position calculation
            # In practice, should use proper SGP4 propagation
            M = sat.mean_anomaly + sat.mean_motion * 360 * (time_offset_seconds / 86400)
            M = M % 360

            # Eccentric anomaly (simplified)
            E = M + sat.eccentricity * np.sin(np.radians(M)) * 180 / np.pi

            # True anomaly
            nu = 2 * np.arctan2(
                np.sqrt(1 + sat.eccentricity) * np.sin(np.radians(E/2)),
                np.sqrt(1 - sat.eccentricity) * np.cos(np.radians(E/2))
            )
            nu = np.degrees(nu)

            # Argument of latitude
            u = sat.arg_perigee + nu

            # Position in orbital plane
            r = sat.altitude_km + EARTH_RADIUS

            # Convert to ECEF
            cos_u = np.cos(np.radians(u))
            sin_u = np.sin(np.radians(u))
            cos_i = np.cos(np.radians(sat.inclination))
            sin_i = np.sin(np.radians(sat.inclination))
            cos_raan = np.cos(np.radians(sat.raan))
            sin_raan = np.sin(np.radians(sat.raan))

            x = r * (cos_raan * cos_u - sin_raan * sin_u * cos_i)
            y = r * (sin_raan * cos_u + cos_raan * sin_u * cos_i)
            z = r * sin_u * sin_i

            # Convert to lat/lon/alt
            lon = np.degrees(np.arctan2(y, x))
            lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
            alt = sat.altitude_km

            positions.append({
                'name': sat.name,
                'norad_id': sat.norad_id,
                'lat': lat,
                'lon': lon,
                'altitude_km': alt,
                'x_km': x,
                'y_km': y,
                'z_km': z,
                'inclination': sat.inclination,
                'period_min': sat.period_minutes
            })

        return positions

    def get_statistics(self) -> Dict:
        """Get statistics about loaded satellite data."""
        stats = {}
        for name, sats in self.satellites.items():
            if not sats:
                continue
            altitudes = [s.altitude_km for s in sats]
            inclinations = [s.inclination for s in sats]
            stats[name] = {
                'count': len(sats),
                'altitude_km': {
                    'min': min(altitudes),
                    'max': max(altitudes),
                    'mean': np.mean(altitudes)
                },
                'inclination_deg': {
                    'min': min(inclinations),
                    'max': max(inclinations),
                    'mean': np.mean(inclinations)
                }
            }
        return stats


if __name__ == '__main__':
    # Test the parser
    parser = TLEParser('data')
    parser.load_all_constellations()

    stats = parser.get_statistics()
    print("Constellation Statistics:")
    for name, info in stats.items():
        print(f"\n{name.upper()}:")
        print(f"  Satellites: {info['count']}")
        print(f"  Altitude: {info['altitude_km']['min']:.1f} - {info['altitude_km']['max']:.1f} km")
        print(f"  Inclination: {info['inclination_deg']['min']:.1f} - {info['inclination_deg']['max']:.1f} deg")

    # Get sample positions
    positions = parser.get_satellite_positions('starlink', num_satellites=5)
    print("\nSample Starlink Positions:")
    for pos in positions:
        print(f"  {pos['name']}: lat={pos['lat']:.2f}, lon={pos['lon']:.2f}, alt={pos['altitude_km']:.1f}km")
