"""
IrisFlow — Modelos do banco de dados
Usuários, licenças e configurações de calibração.
"""

from sqlalchemy import Column, String, Boolean, DateTime, Float, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
import uuid

from api.database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class PlanType(str, enum.Enum):
    familiar = "familiar"
    clinica = "clinica"
    hospitalar = "hospitalar"


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=gen_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    subscription = relationship("Subscription", back_populates="user", uselist=False)
    calibration_config = relationship("CalibrationConfig", back_populates="user", uselist=False)


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    plan = Column(Enum(PlanType), nullable=False)
    stripe_subscription_id = Column(String, unique=True)
    is_active = Column(Boolean, default=False)
    trial_ends_at = Column(DateTime(timezone=True))
    current_period_end = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="subscription")


class CalibrationConfig(Base):
    """
    Armazena configurações de calibração do usuário.
    NOTA: Armazena apenas coeficientes do modelo, NUNCA dados biométricos brutos.
    """
    __tablename__ = "calibration_configs"

    id = Column(String, primary_key=True, default=gen_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, unique=True)
    # Coeficientes da calibração linear (slope + intercept para X e Y)
    x_slope = Column(Float)
    x_intercept = Column(Float)
    y_slope = Column(Float)
    y_intercept = Column(Float)
    dwell_time = Column(Float, default=1.5)   # Tempo de fixação para selecionar (segundos)
    calibration_points = Column(Float, default=0)  # Quantidade de pontos usados
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user = relationship("User", back_populates="calibration_config")
