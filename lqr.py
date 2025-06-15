import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import lqr

dt = 1/240          # Шаг симуляции
g = 9.81            # Сила тяжести
L = 0.8             # Длина невесомого стежня
m = 1               # Масса маятника
b = 0.1             # Коэффициент силы трения

# Зададим небольшое отклонение от целевого положения, чтобы работала линейная модель
th0 = np.pi - 0.1   # Начальное положение в радианах
thd = np.pi         # Целевое положение (вертикально вверх) в радианах

A = np.array([[0, 1],
              [g / L, -b / (m * L ** 2)]])
B = np.array([[0],
              [1 / (m * L ** 2)]])

# Задаем весовые коэффициенты
Q = np.array([[100, 0],
              [0, 1]])
R = 0.1
# Применяем LQR
K, *_ = lqr(A, B, Q, R)
K = -K

physicsClient = p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -g)
planeId = p.loadURDF("plane.urdf")
boxId = p.loadURDF("./simple.urdf.xml", useFixedBase=True)

# Отключаем силы демпфирования
p.changeDynamics(boxId, 1, linearDamping=0, angularDamping=0)
p.changeDynamics(boxId, 2, linearDamping=0, angularDamping=0)

# Устанавливаем начальное положение
p.resetJointState(boxId, 1, th0, 0)

# Выключаем мотор
p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, targetVelocity=0, controlMode=p.VELOCITY_CONTROL, force=0)

maxTime = 5
logTime = np.arange(0, 5, dt)
s = len(logTime)

logThetaSim = np.zeros(s)
logVelSim = np.zeros(s)
logTauSim = np.zeros(s)
idx = 0

for t in logTime:
    th = p.getJointState(boxId, 1)[0]
    vel = p.getJointState(boxId, 1)[1]
    logThetaSim[idx] = th

    # Вычисляем управляющий момент
    tau = K[0,0] * (th - thd) + K[0,1] * vel
    logTauSim[idx] = tau

    # Прикладываем его
    p.setJointMotorControl2(bodyIndex=boxId, jointIndex=1, force=tau, controlMode=p.TORQUE_CONTROL)
    p.stepSimulation()

    vel = p.getJointState(boxId, 1)[1]
    logVelSim[idx] = vel

    idx += 1
p.disconnect()

plt.subplot(3,1,1)
plt.plot(logTime, logThetaSim, 'g', label="Положение маятника (рад)")
plt.plot([logTime[0], logTime[-1]], [thd, thd], 'r--', label="Целевое положение")
plt.grid(True)
plt.legend()

plt.subplot(3,1,2)
plt.plot(logTime, logVelSim, 'g', label="Скорость маятника")
plt.grid(True)
plt.legend()

plt.subplot(3,1,3)
plt.plot(logTime, logTauSim, 'g', label="Управляющий момент")
plt.grid(True)
plt.legend()
plt.show()