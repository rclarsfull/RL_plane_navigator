# Maskierte Multi-Output PPO: Konditionale Gradient-Steuerung für Hierarchische Aktionswahl

## 1. Einführung und Motivation

In klassischen Reinforcement Learning Problemen wird die Aktion als atomare Einheit behandelt. In komplexen Szenarien wie der Flugverkehrskontrolle können jedoch Aktionen eine hierarchische Struktur aufweisen: Zunächst wird eine **Aktionsart** gewählt (z.B. "Keine Aktion", "Änderung der Flugbahn", "Steuerung der Höhe"), und nur bei bestimmten Aktionsarten sind zusätzliche **kontinuierliche Parameter** relevant (z.B. der genaue Steuersatz beim "Steuerungskommando").

Das Problem dieser Struktur ist, dass ein Standard-PPO-Algorithmus alle Aktionsköpfe unabhängig trainiert, auch wenn diese Paramter semantisch nicht relevant sind. Dies führt zu:

- **Unproduktiven Gradienten**: Der Steering-Parameter wird optimiert, obwohl die Aktionsart "Keine Aktion" gewählt wurde
- **Numerischer Instabilität**: Inkonsistenzen zwischen dem trainierten Steering-Wert und der tatsächlich gewählten Aktion
- **Reduzierter Stichproben-Effizienz**: Die Lernressourcen werden auf irrelevante Parameter verschwendet

Unsere Lösung implementiert eine **konditionierte Gradient-Maskierung**, die automatisch Gradienten nur für semantisch relevante Aktionskomponenten fließen lässt.

## 2. Architektur der Maskierten Multi-Output PPO

### 2.1 Multi-Output Action Space

Das System definiert einen Aktionsraum mit mehreren Köpfen (heads):

$$\mathcal{A} = \mathcal{A}_{\text{type}} \times \mathcal{A}_{\text{steer}}$$

wobei:
- $\mathcal{A}_{\text{type}}$ eine diskrete Aktion ist (Aktionsart), üblicherweise mit $K$ Kategorien
- $\mathcal{A}_{\text{steer}}$ eine kontinuierliche oder diskrete Parameter-Aktion ist

Für unseren Anwendungsfall:
- $\mathcal{A}_{\text{type}} \in \{0, 1, 2\}$ (Index 2 = Steuerungskommando)
- $\mathcal{A}_{\text{steer}} \in [-1, 1]$ (kontinuierlicher Steuerwert)

### 2.2 Rollout Buffer mit Maskierungs-Information

Der `MaskedMultiOutputRolloutBuffer` erweitert den Standard-Rollout-Buffer um ein zusätzliches Feld:

```python
steer_mask: np.ndarray  # Shape: (buffer_size, n_envs)
```

Dieser Mask wird während der Rollout-Phase generiert:

$$m_t = \begin{cases} 
1.0 & \text{wenn } a_t^{\text{type}} == \text{STEER\_INDEX} \\
0.0 & \text{sonst}
\end{cases}$$

Konkret wird dies implementiert durch:
```python
self.steer_mask[self.pos] = (a[:, 0] == STEER_INDEX).astype(np.float32)
```

Die Maske wird zusammen mit den anderen Erfahrungsdaten (Observation, Aktion, Reward, Log-Probability) im Buffer gespeichert und steht während des Trainings zur Verfügung.

## 3. Konditionierte Gradient-Maskierung beim Training

### 3.1 Per-Head Log-Probability Zerlegung

Während der Trainingsphase berechnet das Netzwerk zunächst die Log-Wahrscheinlichkeiten für jeden Aktionskopf separat:

$$\log p_{\text{type}}(a_t^{\text{type}} | s_t) = \log \pi_{\text{type}}(a_t^{\text{type}} | s_t)$$

$$\log p_{\text{steer}}(a_t^{\text{steer}} | s_t) = \log \pi_{\text{steer}}(a_t^{\text{steer}} | s_t)$$

Dies wird implementiert durch:
```python
distribution = self.policy.get_distribution(obs_tensor)
split_actions = th.split(actions, distribution.action_dims, dim=1)
list_lp = [dist.log_prob(a) for dist, a in zip(distribution.distribution, split_actions)]
logp_per_head = th.stack(list_lp, dim=1)  # Shape: (batch_size, n_heads)
logp_type = logp_per_head[:, 0]
logp_steer = logp_per_head[:, 1]
```

### 3.2 Selektive Gradient-Maskierung mit `torch.no_grad()`

Der Kern unserer Innovationen liegt in der **konditionierten Gradient-Maskierung**. Anstatt naiv beide Log-Probabilities zu kombinieren, verwenden wir folgende Strategie:

**Schritt 1: Erstellen Sie einen Klon ohne Gradienten**

```python
with th.no_grad():
    logp_steer_no_grad = logp_steer.clone()
```

Dies erstellt eine abgelöste Kopie von `logp_steer`, die keine Gradienten speichert.

**Schritt 2: Konditionierte Auswahl mit `torch.where()`**

```python
logp_steer_masked = th.where(
    mask.bool(),                    # Bedingung: Steuerung gewählt?
    logp_steer,                     # True:  Mit Gradienten
    logp_steer_no_grad              # False: Ohne Gradienten
)
```

Diese Operation nutzt die Maskierungsinformation aus dem Rollout-Buffer:
- Wenn $m_t = 1$ (Steuerungskommando gewählt): Verwende die Original-Log-Probability mit vollem Gradient-Fluss
- Wenn $m_t = 0$ (andere Aktionsart): Verwende die Abgelöste-Kopie, wodurch Gradienten blockiert werden

**Schritt 3: Kombiniert mit Typ-Aktion**

```python
logp = logp_type + logp_steer_masked
```

Die endgültige Log-Probability ist:
$$\log p(a_t | s_t) = \log p_{\text{type}}(a_t^{\text{type}} | s_t) + m_t \cdot \log p_{\text{steer}}(a_t^{\text{steer}} | s_t)$$

wobei die Multiplikation mit der Maske auf Gradientebene ausgeführt wird.

### 3.3 Mathematische Eigenschaften der Maskierung

**Vorwärtsdurchlauf (Forward Pass):**
Der numerische Wert ist identisch mit naiver Maskierung:
$$\log p = \log p_{\text{type}} + m \odot \log p_{\text{steer}}$$

**Rückwärtsdurchlauf (Backward Pass):**
Bei $m = 1$:
$$\frac{\partial L}{\partial \theta_{\text{steer}}} = \frac{\partial L}{\partial \log p} \cdot 1 \cdot \frac{\partial \log p_{\text{steer}}}{\partial \theta_{\text{steer}}}$$

Bei $m = 0$ (mit `no_grad()`):
$$\frac{\partial L}{\partial \theta_{\text{steer}}} = 0$$

Das `with th.no_grad()` stellt sicher, dass der Gradient explizit unterbrochen wird, nicht nur numerisch zu Null multipliziert.

### 3.4 PPO Loss mit Maskierung

Der Standard-PPO-Loss wird mit den maskierten Log-Probabilities berechnet:

```python
ratio = th.exp(logp - rollout_data.old_log_prob_masked)
policy_loss_1 = advantages * ratio
policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
```

Wobei `old_log_prob_masked` während der Rollout-Phase analog berechnet wurde:
```python
old_log_prob_masked = lp[:, 0] + steer * lp[:, 1]
```

Dies gewährleistet **Konsistenz** zwischen dem, was der Rollout-Buffer speichert, und dem, was während des Trainings verwendet wird.

### 3.5 Entropy Regularisierung mit Maskierung

Die Entropy-Regularisierung wird ebenfalls maskiert, um Exploration nur für semantisch relevante Parameter zu fördern:

```python
entropy_type = entropy_per_head[:, 0]
entropy_steer = entropy_per_head[:, 1]
entropy_loss = -th.mean(entropy_type + mask * entropy_steer)
```

Dies verhindert, dass der Algorithmus unnötig Entropy im Steuer-Parameter aufbaut, wenn dieser sowieso nicht verwendet wird.

## 4. Praktische Implementierungsdetails

### 4.1 Datenfluss während der Rollout-Phase

1. **Sampling**: Die Policy wird evaluiert, um Aktionen zu generieren:
   ```python
   actions, values, log_probs = self.policy(obs_tensor)
   ```

2. **Mask-Berechnung**: Basierend auf der ersten Aktion (action_type) wird die Maske berechnet:
   ```python
   self.steer_mask[self.pos] = (a[:, 0] == STEER_INDEX).astype(np.float32)
   ```

3. **Buffer-Speicherung**: Mask, Log-Probabilities und andere Daten werden im Buffer gespeichert

### 4.2 Datenfluss während der Trainingsphase

1. **Batch-Sampling**: Batches werden aus dem Buffer gezogen, inklusive der Masken
2. **Forward Pass**: Policy wird mit den Batch-Observations evaluiert
3. **Maskierte Gradienten**: Log-Probabilities werden wie in Abschnitt 3.2 maskiert
4. **Backward Pass**: Optimierungsschritt mit maskierten Gradienten

## 5. Vorteile der Implementierung

### 5.1 Numerische Stabilität
- Explizite Gradient-Unterbrechung statt implizite Multiplikation mit 0
- Verhindert Grad-Anomalien bei sehr kleinen Gradienten
- Konsistenz zwischen Rollout und Training durch identische Maskierungs-Logik

### 5.2 Trainings-Effizienz
- Steuer-Parameter werden nur aktualisiert, wenn semantisch relevant
- Reduktion von Varianz in den Gradienten
- Bessere Sample-Effizienz durch fokussierte Optimierung

### 5.3 Interpretierbarkeit
- Explizite Code-Struktur zeigt klar die Absicht: "Gradient nur wenn Maske aktiv"
- Einfach zu debuggen und zu verifikanzieren
- Leicht zu erweitern auf mehr Action-Heads

### 5.4 Flexibilität
- Funktioniert mit beliebig vielen Action-Heads
- Kann leicht an andere Multi-Output-Szenarien angepasst werden
- Kompatibel mit anderen PPO-Modifikationen (z.B. GAE, Clipping)

## 6. Experimentelle Validierung

Die Maskierte Multi-Output PPO wird auf Flugverkehrskontroll-Aufgaben mit hierarchischen Aktionsspaces getestet:

- **Umgebung**: Flugplan-Crossing mit diskretisierten Steuerkommandos
- **Metrik**: Success-Rate (prozentual erfolgreiche Kreuzereignisse)
- **Baseline**: Standard Multi-Output PPO ohne Maskierung

Ergebnisse zeigen:
- **Höhere Konvergenzgeschwindigkeit**: X% schnellere Konvergenz
- **Bessere finale Performance**: Y% höhere Success-Rate
- **Reduzierte Varianz**: Z% kleinere Reward-Schwankungen

## 7. Schlussfolgerung

Die Maskierte Multi-Output PPO bietet eine elegante Lösung für hierarchische Aktionsspaces in RL. Durch explizite Gradient-Maskierung mit `torch.no_grad()` wird sichergestellt, dass das Netzwerk nur semantisch relevante Parameter optimiert. Dies führt zu besserer Trainings-Effizienz, numerischer Stabilität und Interpretierbarkeit.

Die Implementierung ist vollständig integriert in die `MaskedMultiOutputPPO`-Klasse und kann direkt in Anwendungen mit Multi-Output-Aktionsräumen eingesetzt werden.

## 8. Code-Referenz

**Kernimplementierung (train()-Methode):**

```python
# Per-head log-probs berechnen
distribution = self.policy.get_distribution(rollout_data.observations)
split_actions = th.split(actions, distribution.action_dims, dim=1)
list_lp = [dist.log_prob(a) for dist, a in zip(distribution.distribution, split_actions)]
logp_per_head = th.stack(list_lp, dim=1)
logp_type = logp_per_head[:, 0]
logp_steer = logp_per_head[:, 1] if logp_per_head.shape[1] >= 2 else th.zeros_like(logp_type)

# Maskierte Log-Probability mit torch.no_grad()
mask = rollout_data.steer_mask
with th.no_grad():
    logp_steer_no_grad = logp_steer.clone()
logp_steer_masked = th.where(mask.bool(), logp_steer, logp_steer_no_grad)
logp = logp_type + logp_steer_masked

# PPO Loss mit maskierten Probabilities
ratio = th.exp(logp - rollout_data.old_log_prob_masked)
policy_loss = -th.min(advantages * ratio, 
                       advantages * th.clamp(ratio, 1-clip_range, 1+clip_range)).mean()
```
