import { useId, useState } from "react";

const layers = ["Curvature", "Thickness", "Elevation"];
function contour(radius: number, variation: number) {
  return (
    Array.from({ length: 121 }, (_, index) => {
      const theta = (index / 120) * Math.PI * 2;
      const wave =
        radius *
        (1 +
          0.045 * Math.sin(theta * 3 + variation) +
          0.06 * Math.cos(theta * 2 - variation));
      return `${index === 0 ? "M" : "L"}${(210 + Math.cos(theta) * wave + 12 * Math.sin(radius / 48)).toFixed(2)},${(190 + Math.sin(theta) * wave * 0.94).toFixed(2)}`;
    }).join(" ") + " Z"
  );
}
export function Topography() {
  const id = useId().replaceAll(":", "");
  const [layer, setLayer] = useState(0);
  return (
    <div className="topography-studio">
      <div className="map-topline">
        <span>Illustrative corneal maps</span>
      </div>
      <svg
        className="topography-map"
        viewBox="0 0 420 380"
        role="img"
        aria-label={`Illustrative ${layers[layer].toLowerCase()} contour map. This is not patient data.`}
      >
        <defs>
          <radialGradient id={`${id}-wash`}>
            <stop stopColor="#a9d594" />
            <stop offset=".45" stopColor="#bedec0" />
            <stop offset="1" stopColor="#e0eee6" />
          </radialGradient>
          <clipPath id={`${id}-clip`}>
            <circle cx="210" cy="190" r="154" />
          </clipPath>
        </defs>
        <circle
          cx="210"
          cy="190"
          r="173"
          fill="none"
          stroke="currentColor"
          strokeOpacity=".12"
        />
        <circle cx="210" cy="190" r="154" fill={`url(#${id}-wash)`} />
        <g clipPath={`url(#${id}-clip)`}>
          {Array.from({ length: 21 }, (_, index) => (
            <path
              key={index}
              d={contour(16 + index * 7.7, layer * 1.7)}
              fill="none"
              stroke="currentColor"
              strokeWidth={index % 4 === 0 ? 1.15 : 0.7}
              opacity={0.24 + index / 65}
            />
          ))}
          <path
            d="M35 190H385 M210 20V360"
            stroke="currentColor"
            strokeOpacity=".22"
            strokeDasharray="3 5"
          />
          <circle cx="210" cy="190" r="3" fill="currentColor" />
        </g>
        {Array.from({ length: 36 }, (_, index) => {
          const theta = (index * Math.PI) / 18;
          return (
            <line
              key={index}
              x1={210 + Math.cos(theta) * 167}
              y1={190 + Math.sin(theta) * 167}
              x2={210 + Math.cos(theta) * (index % 3 === 0 ? 176 : 172)}
              y2={190 + Math.sin(theta) * (index % 3 === 0 ? 176 : 172)}
              stroke="currentColor"
              strokeOpacity=".4"
            />
          );
        })}
        <g fill="currentColor" fontFamily="monospace" fontSize="9" opacity=".6">
          <text x="204" y="8">
            90°
          </text>
          <text x="385" y="194">
            0°
          </text>
          <text x="198" y="379">
            270°
          </text>
          <text x="6" y="194">
            180°
          </text>
        </g>
      </svg>
      <div
        className="map-selector"
        role="group"
        aria-label="Illustration map layer"
      >
        {layers.map((name, index) => (
          <button
            key={name}
            aria-pressed={layer === index}
            className={layer === index ? "active" : ""}
            onClick={() => setLayer(index)}
          >
            {name}
          </button>
        ))}
      </div>
      <div className="map-caption">
        <span>Schematic · no patient data</span>
      </div>
    </div>
  );
}
