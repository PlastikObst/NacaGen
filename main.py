import numpy as np
import matplotlib.pyplot as plt


def calculate_thickness(XL, d):
    """
    Berechnet die Dickenverteilung yd des NACA-Profils gemäß Formel auf S.32

    Parameter:
    XL = x/l : Relative Position auf der Sehnenlänge (0 bis 1)
    d : Nominelle Dicke des Profils

    Returns:
    yd : Dickenverteilung senkrecht zur Sehnenlinie
    """
    yd = (d / 0.2) * (
            0.2969 * np.sqrt(XL) -
            0.1260 * XL -
            0.3516 * XL ** 2 +
            0.2843 * XL ** 3 -
            0.1015 * XL ** 4
    )
    return yd


def calculate_meanline(XL, f, XFL):
    # XL: x/l ist die Position entlang der Sehnenlänge (von 0 bis 1)
    # f: ist die maximale Wölbung (z.B. 0.04 für ein NACA 4xxx)
    # XFL: Position der maximalen Wölbung (z.B. 0.4 für NACA x4xx)

    # Erstelle ein Array mit Nullen in der gleichen Größe wie XL
    ys = np.zeros_like(XL)

    # Berechnung für den vorderen Teil (von der Nase bis zur max. Wölbung):
    # Erstelle eine Maske für alle Punkte wo XL ≤ XFL gilt dafür wird dann true gesetzt
    front = XL <= XFL

    # Berechne die Wölbung für den vorderen Teil mit der Formel:
    # ys = (f/p²) * (2px - x²)
    # wobei: p = XFL, x = XL, f = maximale Wölbung
    ys[front] = (f / XFL ** 2) * (2 * XFL * XL[front] - XL[front] ** 2)

    # Berechnung für den hinteren Teil (von max. Wölbung bis Hinterkante):
    # Erstelle eine Maske für alle Punkte wo XL > XFL gilt
    back = XL > XFL

    # Berechne die Wölbung für den hinteren Teil mit der Formel:
    # ys = (f/(1-p)²) * (1 - 2p + 2px - x²)
    # wobei: p = XFL, x = XL, f = maximale Wölbung
    ys[back] = (f / (1 - XFL) ** 2) * (1 - 2 * XFL + 2 * XFL * XL[back] - XL[back] ** 2)

    return ys


def calculate_airfoil_coordinates(n_points, d, f, XFL):
    """
    Berechnet die Koordinaten des Flügelprofils

    Parameter:
    n_points: Anzahl der Punkte
    d: Dicke (z.B. 0.21 für NACA xx21)
    f: Wölbung (z.B. 0.04 für NACA 4xxx)
    XFL: Position der maximalen Wölbung (z.B. 0.4 für NACA x4xx)

    Returns:
    xu, yu: Koordinaten der Oberseite
    xl, yl: Koordinaten der Unterseite
    """
    # Erzeuge x-Koordinaten mit Cosinus-Verteilung für bessere Auflösung an Vorder- und Hinterkante
    beta = np.linspace(0, np.pi, n_points)
    XL = (1 - np.cos(beta)) / 2

    # Berechne Dickenverteilung
    yd = calculate_thickness(XL, d)

    # Berechne Skelettlinie
    ys = calculate_meanline(XL, f, XFL)

    # Berechne Koordinaten für Ober- und Unterseite
    xu = XL
    yu = ys + yd  # Oberseite = Skelettlinie + Dicke

    xl = XL
    yl = ys - yd  # Unterseite = Skelettlinie - Dicke

    return xu, yu, xl, yl


def save_coordinates_to_csv(xu, yu, xl, yl, filename='airfoil_points.csv'):
    """
    Speichert die Profilkoordinaten in eine CSV-Datei
    Format: x,y,z (z ist immer 0)
    """
    # Erstelle die Punkte-Liste
    points = []

    # Füge Oberseite hinzu (von vorne nach hinten)
    for x, y in zip(xu, yu):
        points.append(f"{x:.6f},{y:.6f},0.000000")

    # Füge Unterseite hinzu (von hinten nach vorne für geschlossenes Profil)
    for x, y in zip(xl[::-1], yl[::-1]):
        points.append(f"{x:.6f},{y:.6f},0.000000")

    # Schreibe in Datei
    with open(filename, 'w') as f:
        f.write('\n'.join(points))


def plot_airfoil(xu, yu, xl, yl):
    """
    Erstellt einen Plot des Profils
    """
    plt.figure(figsize=(15, 10))

    # Hauptplot
    plt.subplot(211)
    plt.plot(xu, yu, 'b-', label='Oberseite')
    plt.plot(xl, yl, 'b-', label='Unterseite')
    plt.plot([0, 1], [0, 0], 'k--', alpha=0.3, label='Sehnenlinie')
    plt.title('NACA Profil')
    plt.xlabel('x/l')
    plt.ylabel('y/l')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')

    # Detailansicht der Hinterkante
    plt.subplot(212)
    plt.plot(xu, yu, 'b-')
    plt.plot(xl, yl, 'b-')
    plt.plot([0, 1], [0, 0], 'k--', alpha=0.3)
    plt.xlim(0.8, 1.02)
    plt.ylim(-0.05, 0.05)
    plt.title('Detailansicht Hinterkante')
    plt.xlabel('x/l')
    plt.ylabel('y/l')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def verify_airfoil(xu, yu, xl, yl, d, f, XFL):
    """
    Überprüft die wichtigsten Eigenschaften des NACA-Profils
    """
    # 1. Überprüfe maximale Dicke
    thickness = np.max(yu - yl)  # Maximale Dicke
    thickness_error = abs(thickness - d) / d * 100
    print(f"Maximale Dicke: {thickness:.4f} (Soll: {d:.4f}, Fehler: {thickness_error:.2f}%)")

    # 2. Überprüfe maximale Wölbung
    camber = (yu + yl) / 2  # Mittellinie
    max_camber = np.max(camber)
    max_camber_error = abs(max_camber - f) / f * 100
    print(f"Maximale Wölbung: {max_camber:.4f} (Soll: {f:.4f}, Fehler: {max_camber_error:.2f}%)")

    # 3. Überprüfe Position der maximalen Wölbung
    camber_position = xu[np.argmax(camber)]
    position_error = abs(camber_position - XFL) / XFL * 100
    print(f"Position max. Wölbung: {camber_position:.4f} (Soll: {XFL:.4f}, Fehler: {position_error:.2f}%)")

    # 4. Überprüfe Hinterkante
    trailing_edge_gap = np.sqrt((xu[-1] - xl[-1]) ** 2 + (yu[-1] - yl[-1]) ** 2)
    print(f"Hinterkanten-Abstand: {trailing_edge_gap:.6f}")

    return all([
        thickness_error < 1,  # weniger als 1% Fehler
        max_camber_error < 1,
        position_error < 5,
        trailing_edge_gap < 0.001
    ])


# Hauptprogramm
if __name__ == "__main__":
    # Parameter für NACA 4421
    n_points = 1000
    d = 0.21  # Dicke 21%
    f = 0.04  # Wölbung 4%
    XFL = 0.4  # Position der max. Wölbung 40%

    # Berechne Koordinaten
    xu, yu, xl, yl = calculate_airfoil_coordinates(n_points, d, f, XFL)

    # Speichere Koordinaten
    save_coordinates_to_csv(xu, yu, xl, yl)

    # Zeige Plot
    plot_airfoil(xu, yu, xl, yl)

    #verify
    verify_airfoil(xu, yu, xl, yl, d, f, XFL)

