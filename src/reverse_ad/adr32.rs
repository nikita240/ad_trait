use spin::RwLock;

use super::adr::NodeIdx;
use crate::{ADNumMode, ADNumType, AD, F64};
use alloc::vec::Vec;
use alloc::{format, vec};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
#[cfg(feature = "bevy")]
use bevy_reflect::Reflect;
use core::cmp::Ordering;
use core::fmt;
use core::fmt::{Debug, Display, Formatter};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use nalgebra::{DefaultAllocator, Dim, DimName, Matrix, OPoint, RawStorageMut};
use ndarray::{ArrayBase, Dimension, OwnedRepr, ScalarOperand};
use num_traits::{Bounded, FromPrimitive, Num, One, Signed, Zero};
use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use simba::scalar::{ComplexField, Field, RealField, SubsetOf};
use simba::simd::{PrimitiveSimdValue, SimdValue};
use tinyvec::{tiny_vec, TinyVec};

/// A type for Reverse-mode Automatic Differentiation using f32 storage.
///
/// `adr32` stores its current value as f32 and a reference to its position (node index)
/// in a global computation graph. This allows for computing gradients by rebuilding
/// the chain of operations and backpropagating adjoints.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
#[cfg_attr(feature = "bevy", derive(Reflect))]
#[cfg_attr(feature = "bevy", reflect(from_reflect = false))]
pub struct adr32 {
    /// The primary value.
    value: f32,
    /// The index of the node representing this value in the computation graph.
    #[cfg_attr(feature = "bevy", reflect(ignore))]
    node_idx: NodeIdx,
}
impl Debug for adr32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        f.write_str("adr32 { ").expect("error");
        f.write_str(&format!("value: {:?}, ", self.value()))
            .expect("error");
        f.write_str(&format!("node_idx: {:?}", self.node_idx))
            .expect("error");
        f.write_str(" }").expect("error");

        Ok(())
    }
}
impl Default for adr32 {
    fn default() -> Self {
        Self::zero()
    }
}
impl adr32 {
    /// Creates a new variable in the global computation graph.
    ///
    /// # Arguments
    /// * `value` - The initial value.
    /// * `reset_computation_graph` - If true, the global graph will be cleared before adding this variable.
    pub fn new_variable(value: f32, reset_computation_graph: bool) -> Self {
        if reset_computation_graph {
            GlobalComputationGraph32::get().reset();
        }
        GlobalComputationGraph32::get().spawn_value(value)
    }
    #[inline]
    pub fn value(&self) -> f32 {
        self.value
    }
    #[inline]
    pub fn is_constant(&self) -> bool {
        match self.node_idx {
            NodeIdx::Constant => true,
            _ => false,
        }
    }
    /// Initiates a backward pass from this node to compute gradients (adjoints)
    /// for all parent nodes in the computation graph.
    pub fn get_backwards_mode_grad(&self) -> BackwardsModeGradOutput32 {
        GlobalComputationGraph32::get().get_backwards_mode_grad(self.node_idx)
    }
}

impl Serialize for adr32 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("adr32", 1)?;
        state.serialize_field("value", &self.value)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for adr32 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            Value,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                        formatter.write_str("value or tangent")
                    }

                    fn visit_str<E: de::Error>(self, value: &str) -> Result<Field, E> {
                        match value {
                            "value" => Ok(Field::Value),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct Adr32Visitor;

        impl<'de> Visitor<'de> for Adr32Visitor {
            type Value = adr32;

            fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                formatter.write_str("struct adr32")
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<adr32, V::Error> {
                let mut value: Option<f32> = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Value => {
                            if value.is_some() {
                                return Err(de::Error::duplicate_field("value"));
                            }
                            value = Some(map.next_value()?);
                        }
                    }
                }

                let value = value.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(adr32 {
                    value,
                    node_idx: NodeIdx::Constant,
                })
            }
        }

        const FIELDS: &'static [&'static str] = &["value"];
        deserializer.deserialize_struct("adr32", FIELDS, Adr32Visitor)
    }
}

#[derive(Clone, Debug)]
pub struct BackwardsModeGradOutput32 {
    adjoints: Vec<f32>,
}
impl BackwardsModeGradOutput32 {
    pub fn wrt(&self, v: &adr32) -> f32 {
        self.adjoints[v.node_idx.get_idx()]
    }
}

impl AD for adr32 {
    fn constant(constant: f64) -> Self {
        return adr32 {
            value: constant as f32,
            node_idx: NodeIdx::Constant,
        };
    }

    fn to_constant(&self) -> f64 {
        self.value() as f64
    }

    fn ad_num_mode() -> ADNumMode {
        ADNumMode::ReverseAD
    }

    fn ad_num_type() -> ADNumType {
        ADNumType::ADR32
    }

    fn add_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) + arg2
    }

    fn sub_l_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) - arg2
    }

    fn sub_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 - Self::constant(arg2)
    }

    fn mul_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) * arg2
    }

    fn div_l_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) / arg2
    }

    fn div_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 / Self::constant(arg2)
    }

    fn rem_l_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) % arg2
    }

    fn rem_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 % Self::constant(arg2)
    }

    fn mul_by_nalgebra_matrix<
        R: Clone + Dim,
        C: Clone + Dim,
        S: Clone + RawStorageMut<Self, R, C>,
    >(
        &self,
        other: Matrix<Self, R, C, S>,
    ) -> Matrix<Self, R, C, S> {
        *self * other
    }

    fn mul_by_nalgebra_matrix_ref<
        'a,
        R: Clone + Dim,
        C: Clone + Dim,
        S: Clone + RawStorageMut<Self, R, C>,
    >(
        &'a self,
        other: &'a Matrix<Self, R, C, S>,
    ) -> Matrix<Self, R, C, S> {
        *self * other
    }

    fn mul_by_ndarray_matrix_ref<D: Dimension>(
        &self,
        other: &ArrayBase<OwnedRepr<Self>, D>,
    ) -> ArrayBase<OwnedRepr<Self>, D> {
        other * *self
    }
}

impl ScalarOperand for adr32 {}

/// The default number of nodes to pre-allocate in the computation graph.
/// Each node is approximately 40-48 bytes on 32-bit targets with f32 storage.
pub const DEFAULT_PREALLOCATED_NODES_32: usize = 512;

/// The number of additional nodes to allocate when the graph runs out of space.
const GROWTH_CHUNK_SIZE_32: usize = 256;

#[derive(Debug)]
pub struct ComputationGraph32 {
    add_idx: RwLock<usize>,
    nodes: RwLock<Vec<ComputationGraphNode32>>,
}
impl ComputationGraph32 {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            add_idx: RwLock::new(0),
            nodes: RwLock::new(vec![]),
        }
    }
    #[allow(dead_code)]
    fn new_preallocated(num_to_preallocate: usize) -> Self {
        Self {
            add_idx: RwLock::new(0),
            nodes: RwLock::new(vec![ComputationGraphNode32::default(); num_to_preallocate]),
        }
    }
    fn reset(&self) {
        *self.add_idx.write() = 0;
    }
    pub fn get_backwards_mode_grad(&self, node_idx_enum: NodeIdx) -> BackwardsModeGradOutput32 {
        let nodes = self.nodes.read();
        let add_idx = *self.add_idx.read();
        let l = add_idx;
        let mut adjoints = vec![0.0f32; l];
        match node_idx_enum {
            NodeIdx::Constant => {
                return BackwardsModeGradOutput32 { adjoints };
            }
            NodeIdx::Idx(node_idx) => {
                adjoints[node_idx] = 1.0;
            }
        }

        for node_idx in (0..l).rev() {
            let node = &nodes[node_idx];
            let parent_adjoints = node
                .node_type
                .get_derivatives_wrt_parents(node.parent_0, node.parent_1);
            if parent_adjoints.len() == 1 {
                let curr_adjoint = adjoints[node_idx];
                let parent_0_idx = node.parent_0_idx.unwrap();
                if parent_0_idx != NodeIdx::Constant {
                    adjoints[parent_0_idx.get_idx()] += curr_adjoint * parent_adjoints[0];
                }
            } else if parent_adjoints.len() == 2 {
                let curr_adjoint = adjoints[node_idx];
                let parent_0_idx = node.parent_0_idx.unwrap();
                let parent_1_idx = node.parent_1_idx.unwrap();
                if parent_0_idx != NodeIdx::Constant {
                    adjoints[parent_0_idx.get_idx()] += curr_adjoint * parent_adjoints[0];
                }
                if parent_1_idx != NodeIdx::Constant {
                    adjoints[parent_1_idx.get_idx()] += curr_adjoint * parent_adjoints[1];
                }
            }
        }

        BackwardsModeGradOutput32 { adjoints }
    }
    #[inline(always)]
    fn spawn_variable(&self, value: f32) -> adr32 {
        let mut nodes = self.nodes.write();
        let mut add_idx = self.add_idx.write();
        let node_idx = *add_idx;
        let l = nodes.len();

        let node = ComputationGraphNode32 {
            node_idx,
            node_type: NodeType32::Constant,
            value,
            parent_0: None,
            parent_1: None,
            parent_0_idx: None,
            parent_1_idx: None,
        };

        if node_idx >= l {
            nodes.push(node);
            for _ in 0..GROWTH_CHUNK_SIZE_32 {
                nodes.push(ComputationGraphNode32::default());
            }
        } else {
            nodes[node_idx] = node;
        }

        let out = adr32 {
            value,
            node_idx: NodeIdx::Idx(node_idx),
        };

        *add_idx += 1;

        out
    }
    #[inline(always)]
    fn add_node(
        &self,
        node_type: NodeType32,
        value: f32,
        parent_0: Option<f32>,
        parent_1: Option<f32>,
        parent_0_idx: Option<NodeIdx>,
        parent_1_idx: Option<NodeIdx>,
    ) -> adr32 {
        if parent_0_idx.is_some() {
            if parent_1_idx.is_some() {
                if parent_0_idx.unwrap() == NodeIdx::Constant
                    && parent_1_idx.unwrap() == NodeIdx::Constant
                {
                    return adr32 {
                        value,
                        node_idx: NodeIdx::Constant,
                    };
                }
            } else {
                if parent_0_idx.unwrap() == NodeIdx::Constant {
                    return adr32 {
                        value,
                        node_idx: NodeIdx::Constant,
                    };
                }
            }
        }

        let mut nodes = self.nodes.write();
        let mut add_idx = self.add_idx.write();
        let node_idx = *add_idx;
        let l = nodes.len();
        if node_idx >= l {
            nodes.push(ComputationGraphNode32 {
                node_idx,
                node_type,
                value,
                parent_0,
                parent_1,
                parent_0_idx,
                parent_1_idx,
            });

            for _ in 0..GROWTH_CHUNK_SIZE_32 {
                nodes.push(ComputationGraphNode32::default());
            }
        } else {
            nodes[*add_idx] = ComputationGraphNode32 {
                node_idx,
                node_type,
                value,
                parent_0,
                parent_1,
                parent_0_idx,
                parent_1_idx,
            }
        }

        let out = adr32 {
            value,
            node_idx: NodeIdx::Idx(node_idx),
        };

        *add_idx += 1;

        return out;
    }
}

#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
pub struct ComputationGraphNode32 {
    node_idx: usize,
    node_type: NodeType32,
    value: f32,
    parent_0: Option<f32>,
    parent_1: Option<f32>,
    parent_0_idx: Option<NodeIdx>,
    parent_1_idx: Option<NodeIdx>,
}

#[derive(Clone, Debug, Copy, Default)]
pub enum NodeType32 {
    #[default]
    Constant,
    Add,
    Mul,
    Sub,
    Div,
    Neg,
    Abs,
    Signum,
    Max,
    Min,
    Atan2,
    Floor,
    Ceil,
    Round,
    Trunc,
    Fract,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    Log,
    Sqrt,
    Exp,
    Powf,
}
impl NodeType32 {
    fn get_derivatives_wrt_parents(
        &self,
        parent_0: Option<f32>,
        parent_1: Option<f32>,
    ) -> TinyVec<[f32; 2]> {
        return match self {
            NodeType32::Constant => {
                tiny_vec!([f32; 2])
            }
            NodeType32::Add => {
                tiny_vec!([f32; 2] => 1.0, 1.0)
            }
            NodeType32::Mul => {
                tiny_vec!([f32; 2] => parent_1.unwrap(), parent_0.unwrap())
            }
            NodeType32::Sub => {
                tiny_vec!([f32; 2] => 1.0, -1.0)
            }
            NodeType32::Div => {
                tiny_vec!([f32; 2] => 1.0/parent_1.unwrap(), -parent_0.unwrap()/(parent_1.unwrap()*parent_1.unwrap()))
            }
            NodeType32::Neg => {
                tiny_vec!([f32; 2] => -1.0)
            }
            NodeType32::Abs => {
                let val = parent_0.unwrap();
                if val >= 0.0 {
                    tiny_vec!([f32; 2] => 1.0)
                } else {
                    tiny_vec!([f32; 2] => -1.0)
                }
            }
            NodeType32::Signum => {
                tiny_vec!([f32; 2] => 0.0)
            }
            NodeType32::Max => {
                if parent_0.unwrap() >= parent_1.unwrap() {
                    tiny_vec!([f32; 2] => 1.0, 0.0)
                } else {
                    tiny_vec!([f32; 2] => 0.0, 1.0)
                }
            }
            NodeType32::Min => {
                if parent_0.unwrap() <= parent_1.unwrap() {
                    tiny_vec!([f32; 2] => 1.0, 0.0)
                } else {
                    tiny_vec!([f32; 2] => 0.0, 1.0)
                }
            }
            NodeType32::Atan2 => {
                let lhs = parent_0.unwrap();
                let rhs = parent_1.unwrap();
                tiny_vec!([f32; 2] => rhs/(lhs*lhs + rhs*rhs), -lhs/(lhs*lhs + rhs*rhs))
            }
            NodeType32::Floor => {
                tiny_vec!([f32; 2] => 0.0)
            }
            NodeType32::Ceil => {
                tiny_vec!([f32; 2] => 0.0)
            }
            NodeType32::Round => {
                tiny_vec!([f32; 2] => 0.0)
            }
            NodeType32::Trunc => {
                tiny_vec!([f32; 2] => 0.0)
            }
            NodeType32::Fract => {
                tiny_vec!([f32; 2] => 1.0)
            }
            NodeType32::Sin => {
                tiny_vec!([f32; 2] => parent_0.unwrap().cos())
            }
            NodeType32::Cos => {
                tiny_vec!([f32; 2] => (-parent_0.unwrap()).sin())
            }
            NodeType32::Tan => {
                let c = parent_0.unwrap().cos();
                tiny_vec!([f32; 2] => 1.0 / (c*c))
            }
            NodeType32::Asin => {
                tiny_vec!([f32; 2] => 1.0 / (1.0f32 - parent_0.unwrap() * parent_0.unwrap()).sqrt())
            }
            NodeType32::Acos => {
                tiny_vec!([f32; 2] => -1.0 / (1.0f32 - parent_0.unwrap() * parent_0.unwrap()).sqrt())
            }
            NodeType32::Atan => {
                tiny_vec!([f32; 2] => 1.0/(parent_0.unwrap()*parent_0.unwrap() + 1.0))
            }
            NodeType32::Sinh => {
                tiny_vec!([f32; 2] => parent_0.unwrap().cosh())
            }
            NodeType32::Cosh => {
                tiny_vec!([f32; 2] => parent_0.unwrap().sinh())
            }
            NodeType32::Tanh => {
                let c = parent_0.unwrap().cosh();
                tiny_vec!([f32; 2] => 1.0 / (c*c))
            }
            NodeType32::Asinh => {
                let lhs = parent_0.unwrap();
                tiny_vec!([f32; 2] => 1.0/(lhs*lhs + 1.0).sqrt())
            }
            NodeType32::Acosh => {
                let lhs = parent_0.unwrap();
                tiny_vec!([f32; 2] => 1.0/((lhs*lhs - 1.0f32).sqrt()) )
            }
            NodeType32::Atanh => {
                let lhs = parent_0.unwrap();
                tiny_vec!([f32; 2] => 1.0/(1.0 - lhs*lhs))
            }
            NodeType32::Log => {
                let lhs = parent_0.unwrap();
                let rhs = parent_1.unwrap();
                let ln_rhs = rhs.ln();
                let ln_lhs = lhs.ln();
                tiny_vec!([f32; 2] => 1.0/(lhs * ln_rhs), -ln_lhs / (rhs * ln_rhs * ln_rhs))
            }
            NodeType32::Sqrt => {
                let lhs = parent_0.unwrap();
                let tmp = if lhs == 0.0 { 0.0001f32 } else { lhs };
                tiny_vec!([f32; 2] => 1.0/(2.0*tmp.sqrt()))
            }
            NodeType32::Exp => {
                tiny_vec!([f32; 2] => parent_0.unwrap().exp())
            }
            NodeType32::Powf => {
                let lhs = parent_0.unwrap();
                let rhs = parent_1.unwrap();
                let tmp = if lhs == 0.0 { 0.0001f32 } else { lhs };
                tiny_vec!([f32; 2] => rhs * lhs.powf(rhs - 1.0), lhs.powf(rhs) * tmp.ln())
            }
        };
    }
}

static GRAPH32: spin::Lazy<ComputationGraph32> =
    spin::Lazy::new(|| ComputationGraph32::new_preallocated(DEFAULT_PREALLOCATED_NODES_32));

pub struct GlobalComputationGraph32;
impl GlobalComputationGraph32 {
    pub fn reset(&self) {
        GRAPH32.reset()
    }
    pub fn spawn_value(&self, value: f32) -> adr32 {
        GRAPH32.spawn_variable(value)
    }
    pub fn get() -> GlobalComputationGraph32 {
        GlobalComputationGraph32
    }
    pub fn num_nodes(&self) -> usize {
        *GRAPH32.add_idx.read()
    }
    pub fn add_node(
        &self,
        node_type: NodeType32,
        value: f32,
        parent_0: Option<f32>,
        parent_1: Option<f32>,
        parent_0_idx: Option<NodeIdx>,
        parent_1_idx: Option<NodeIdx>,
    ) -> adr32 {
        GRAPH32.add_node(
            node_type,
            value,
            parent_0,
            parent_1,
            parent_0_idx,
            parent_1_idx,
        )
    }
    pub fn get_backwards_mode_grad(&self, node_idx: NodeIdx) -> BackwardsModeGradOutput32 {
        GRAPH32.get_backwards_mode_grad(node_idx)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl Add<F64> for adr32 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F64) -> Self::Output {
        AD::add_scalar(rhs.0, self)
    }
}

impl AddAssign<F64> for adr32 {
    #[inline]
    fn add_assign(&mut self, rhs: F64) {
        *self = *self + rhs;
    }
}

impl Mul<F64> for adr32 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F64) -> Self::Output {
        AD::mul_scalar(rhs.0, self)
    }
}

impl MulAssign<F64> for adr32 {
    #[inline]
    fn mul_assign(&mut self, rhs: F64) {
        *self = *self * rhs;
    }
}

impl Sub<F64> for adr32 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F64) -> Self::Output {
        AD::sub_r_scalar(self, rhs.0)
    }
}

impl SubAssign<F64> for adr32 {
    #[inline]
    fn sub_assign(&mut self, rhs: F64) {
        *self = *self - rhs;
    }
}

impl Div<F64> for adr32 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: F64) -> Self::Output {
        AD::div_r_scalar(self, rhs.0)
    }
}

impl DivAssign<F64> for adr32 {
    #[inline]
    fn div_assign(&mut self, rhs: F64) {
        *self = *self / rhs;
    }
}

impl Rem<F64> for adr32 {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: F64) -> Self::Output {
        AD::rem_r_scalar(self, rhs.0)
    }
}

impl RemAssign<F64> for adr32 {
    #[inline]
    fn rem_assign(&mut self, rhs: F64) {
        *self = *self % rhs;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl Add<Self> for adr32 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let out_value = self.value() + rhs.value();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Add,
            out_value,
            Some(self.value()),
            Some(rhs.value()),
            Some(self.node_idx),
            Some(rhs.node_idx),
        )
    }
}
impl AddAssign<Self> for adr32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Mul<Self> for adr32 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let out_value = self.value() * rhs.value();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Mul,
            out_value,
            Some(self.value()),
            Some(rhs.value()),
            Some(self.node_idx),
            Some(rhs.node_idx),
        )
    }
}
impl MulAssign<Self> for adr32 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Sub<Self> for adr32 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let out_value = self.value() - rhs.value();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Sub,
            out_value,
            Some(self.value()),
            Some(rhs.value()),
            Some(self.node_idx),
            Some(rhs.node_idx),
        )
    }
}
impl SubAssign<Self> for adr32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Div<Self> for adr32 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let out_value = self.value() / rhs.value();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Div,
            out_value,
            Some(self.value()),
            Some(rhs.value()),
            Some(self.node_idx),
            Some(rhs.node_idx),
        )
    }
}
impl DivAssign<Self> for adr32 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Rem<Self> for adr32 {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        self - ComplexField::floor(self / rhs) * rhs
    }
}
impl RemAssign<Self> for adr32 {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl Neg for adr32 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        let out_value = self.value().neg();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Neg,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl PartialEq for adr32 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}

impl PartialOrd for adr32 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value().partial_cmp(&other.value())
    }
}

impl Display for adr32 {
    fn fmt(&self, f: &mut Formatter) -> core::fmt::Result {
        f.write_str(&format!("{:?}", self)).expect("error");
        Ok(())
    }
}

impl From<f64> for adr32 {
    fn from(value: f64) -> Self {
        adr32::constant(value)
    }
}
impl Into<f64> for adr32 {
    fn into(self) -> f64 {
        self.value() as f64
    }
}
impl From<f32> for adr32 {
    fn from(value: f32) -> Self {
        adr32::constant(value as f64)
    }
}
impl Into<f32> for adr32 {
    fn into(self) -> f32 {
        self.value()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl UlpsEq for adr32 {
    fn default_max_ulps() -> u32 {
        unimplemented!("take the time to figure this out.")
    }

    fn ulps_eq(&self, _other: &Self, _epsilon: Self::Epsilon, _max_ulps: u32) -> bool {
        unimplemented!("take the time to figure this out.")
    }
}

impl AbsDiffEq for adr32 {
    type Epsilon = Self;

    fn default_epsilon() -> Self::Epsilon {
        Self::constant(0.000000001)
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let diff = *self - *other;
        if diff.value().abs() < epsilon.value() {
            true
        } else {
            false
        }
    }
}

impl RelativeEq for adr32 {
    fn default_max_relative() -> Self::Epsilon {
        Self::constant(0.000000001)
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        _max_relative: Self::Epsilon,
    ) -> bool {
        let diff = *self - *other;
        if diff.value().abs() < epsilon.value() {
            true
        } else {
            false
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl SimdValue for adr32 {
    const LANES: usize = 4;
    type Element = Self;
    type SimdBool = bool;

    fn splat(val: Self::Element) -> Self {
        val
    }

    fn extract(&self, _: usize) -> Self::Element {
        *self
    }

    unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
        *self
    }

    fn replace(&mut self, _: usize, val: Self::Element) {
        *self = val
    }

    unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
        *self = val
    }

    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        if cond {
            self
        } else {
            other
        }
    }
}

impl<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<adr32, R, C>>
    Mul<Matrix<adr32, R, C, S>> for adr32
{
    type Output = Matrix<Self, R, C, S>;

    fn mul(self, rhs: Matrix<Self, R, C, S>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

impl<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<adr32, R, C>>
    Mul<&Matrix<adr32, R, C, S>> for adr32
{
    type Output = Matrix<Self, R, C, S>;

    fn mul(self, rhs: &Matrix<Self, R, C, S>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

impl<D: DimName> Mul<OPoint<adr32, D>> for adr32
where
    DefaultAllocator: nalgebra::allocator::Allocator<adr32, D>,
    DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    type Output = OPoint<adr32, D>;

    fn mul(self, rhs: OPoint<adr32, D>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

impl<D: DimName> Mul<&OPoint<adr32, D>> for adr32
where
    DefaultAllocator: nalgebra::allocator::Allocator<adr32, D>,
    DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    type Output = OPoint<adr32, D>;

    fn mul(self, rhs: &OPoint<adr32, D>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl Zero for adr32 {
    #[inline]
    fn zero() -> Self {
        return Self::constant(0.0);
    }

    fn is_zero(&self) -> bool {
        return self.value() == 0.0;
    }
}

impl One for adr32 {
    #[inline]
    fn one() -> Self {
        Self::constant(1.0)
    }
}

impl Num for adr32 {
    type FromStrRadixErr = ();

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let val = f64::from_str_radix(str, radix).expect("error");
        Ok(Self::constant(val))
    }
}

impl Signed for adr32 {
    fn abs(&self) -> Self {
        let out_value = self.value().abs();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Abs,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    fn abs_sub(&self, other: &Self) -> Self {
        return if self.value() <= other.value() {
            Self::constant(0.0)
        } else {
            *self - *other
        };
    }

    fn signum(&self) -> Self {
        let out_value = self.value().signum();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Signum,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    fn is_positive(&self) -> bool {
        return self.value() > 0.0;
    }

    fn is_negative(&self) -> bool {
        return self.value() < 0.0;
    }
}

impl FromPrimitive for adr32 {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::constant(n as f64))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::constant(n as f64))
    }
}

impl Bounded for adr32 {
    fn min_value() -> Self {
        Self::constant(f32::MIN as f64)
    }

    fn max_value() -> Self {
        Self::constant(f32::MAX as f64)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl RealField for adr32 {
    fn is_sign_positive(&self) -> bool {
        return self.is_positive();
    }

    fn is_sign_negative(&self) -> bool {
        return self.is_negative();
    }

    fn copysign(self, sign: Self) -> Self {
        return if sign.is_positive() {
            ComplexField::abs(self)
        } else {
            -ComplexField::abs(self)
        };
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        let out_value = self.value().max(other.value());
        GlobalComputationGraph32::get().add_node(
            NodeType32::Max,
            out_value,
            Some(self.value()),
            Some(other.value()),
            Some(self.node_idx),
            Some(other.node_idx),
        )
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        let out_value = self.value().min(other.value());
        GlobalComputationGraph32::get().add_node(
            NodeType32::Min,
            out_value,
            Some(self.value()),
            Some(other.value()),
            Some(self.node_idx),
            Some(other.node_idx),
        )
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        assert!(min.value() <= max.value());
        return RealField::min(RealField::max(self, min), max);
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        let out_value = self.value().atan2(other.value());
        GlobalComputationGraph32::get().add_node(
            NodeType32::Atan2,
            out_value,
            Some(self.value()),
            Some(other.value()),
            Some(self.node_idx),
            Some(other.node_idx),
        )
    }

    #[inline]
    fn min_value() -> Option<Self> {
        Some(Self::constant(f32::MIN as f64))
    }

    #[inline]
    fn max_value() -> Option<Self> {
        Some(Self::constant(f32::MAX as f64))
    }

    #[inline]
    fn pi() -> Self {
        Self::constant(core::f32::consts::PI as f64)
    }

    #[inline]
    fn two_pi() -> Self {
        Self::constant(2.0 * core::f32::consts::PI as f64)
    }

    #[inline]
    fn frac_pi_2() -> Self {
        Self::constant(core::f32::consts::FRAC_PI_2 as f64)
    }

    #[inline]
    fn frac_pi_3() -> Self {
        Self::constant(core::f32::consts::FRAC_PI_3 as f64)
    }

    #[inline]
    fn frac_pi_4() -> Self {
        Self::constant(core::f32::consts::FRAC_PI_4 as f64)
    }

    #[inline]
    fn frac_pi_6() -> Self {
        Self::constant(core::f32::consts::FRAC_PI_6 as f64)
    }

    #[inline]
    fn frac_pi_8() -> Self {
        Self::constant(core::f32::consts::FRAC_PI_8 as f64)
    }

    #[inline]
    fn frac_1_pi() -> Self {
        Self::constant(core::f32::consts::FRAC_1_PI as f64)
    }

    #[inline]
    fn frac_2_pi() -> Self {
        Self::constant(core::f32::consts::FRAC_2_PI as f64)
    }

    #[inline]
    fn frac_2_sqrt_pi() -> Self {
        Self::constant(core::f32::consts::FRAC_2_SQRT_PI as f64)
    }

    #[inline]
    fn e() -> Self {
        Self::constant(core::f32::consts::E as f64)
    }

    #[inline]
    fn log2_e() -> Self {
        Self::constant(core::f32::consts::LOG2_E as f64)
    }

    #[inline]
    fn log10_e() -> Self {
        Self::constant(core::f32::consts::LOG10_E as f64)
    }

    #[inline]
    fn ln_2() -> Self {
        Self::constant(core::f32::consts::LN_2 as f64)
    }

    #[inline]
    fn ln_10() -> Self {
        Self::constant(core::f32::consts::LN_10 as f64)
    }
}

impl ComplexField for adr32 {
    type RealField = Self;

    fn from_real(re: Self::RealField) -> Self {
        re.clone()
    }

    fn real(self) -> <Self as ComplexField>::RealField {
        self.clone()
    }

    fn imaginary(self) -> Self::RealField {
        Self::zero()
    }

    fn modulus(self) -> Self::RealField {
        return ComplexField::abs(self);
    }

    fn modulus_squared(self) -> Self::RealField {
        self * self
    }

    fn argument(self) -> Self::RealField {
        unimplemented!();
    }

    fn norm1(self) -> Self::RealField {
        return ComplexField::abs(self);
    }

    fn scale(self, factor: Self::RealField) -> Self {
        return self * factor;
    }

    fn unscale(self, factor: Self::RealField) -> Self {
        return self / factor;
    }

    #[inline]
    fn floor(self) -> Self {
        let out_value = self.value().floor();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Floor,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn ceil(self) -> Self {
        let out_value = self.value().ceil();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Ceil,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn round(self) -> Self {
        let out_value = self.value().round();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Round,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn trunc(self) -> Self {
        let out_value = self.value().trunc();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Trunc,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn fract(self) -> Self {
        let out_value = self.value().fract();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Fract,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        return (self * a) + b;
    }

    #[inline]
    fn abs(self) -> Self::RealField {
        <Self as Signed>::abs(&self)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self::RealField {
        return ComplexField::sqrt(ComplexField::powi(self, 2) + ComplexField::powi(other, 2));
    }

    #[inline]
    fn recip(self) -> Self {
        return Self::constant(1.0) / self;
    }

    #[inline]
    fn conjugate(self) -> Self {
        return self;
    }

    #[inline]
    fn sin(self) -> Self {
        let out_value = self.value().sin();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Sin,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn cos(self) -> Self {
        let out_value = self.value().cos();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Cos,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        return (ComplexField::sin(self), ComplexField::cos(self));
    }

    #[inline]
    fn tan(self) -> Self {
        let out_value = self.value().tan();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Tan,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn asin(self) -> Self {
        let out_value = self.value().asin();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Asin,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn acos(self) -> Self {
        let out_value = self.value().acos();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Acos,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn atan(self) -> Self {
        let out_value = self.value().atan();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Atan,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn sinh(self) -> Self {
        let out_value = self.value().sinh();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Sinh,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn cosh(self) -> Self {
        let out_value = self.value().cosh();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Cosh,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn tanh(self) -> Self {
        let out_value = self.value().tanh();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Tanh,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn asinh(self) -> Self {
        let out_value = self.value().asinh();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Asinh,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn acosh(self) -> Self {
        let out_value = self.value().acosh();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Acosh,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn atanh(self) -> Self {
        let out_value = self.value().atanh();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Atanh,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn log(self, base: Self::RealField) -> Self {
        let out_value = self.value().log(base.value());
        GlobalComputationGraph32::get().add_node(
            NodeType32::Log,
            out_value,
            Some(self.value()),
            Some(base.value()),
            Some(self.node_idx),
            Some(base.node_idx),
        )
    }

    #[inline]
    fn log2(self) -> Self {
        return ComplexField::log(self, Self::constant(2.0));
    }

    #[inline]
    fn log10(self) -> Self {
        return ComplexField::log(self, Self::constant(10.0));
    }

    #[inline]
    fn ln(self) -> Self {
        return ComplexField::log(self, Self::constant(core::f32::consts::E as f64));
    }

    #[inline]
    fn ln_1p(self) -> Self {
        ComplexField::ln(Self::constant(1.0) + self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        let out_value = self.value().sqrt();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Sqrt,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn exp(self) -> Self {
        let out_value = self.value().exp();
        GlobalComputationGraph32::get().add_node(
            NodeType32::Exp,
            out_value,
            Some(self.value()),
            None,
            Some(self.node_idx),
            None,
        )
    }

    #[inline]
    fn exp2(self) -> Self {
        ComplexField::powf(Self::constant(2.0), self)
    }

    #[inline]
    fn exp_m1(self) -> Self {
        return ComplexField::exp(self) - Self::constant(1.0);
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        return ComplexField::powf(self, Self::constant(n as f64));
        // // Use repeated multiplication instead of powf to handle negative bases
        // // (powf uses ln internally which is undefined for negative values)
        // if n == 0 {
        //     return Self::one();
        // }
        // let mut result = self;
        // let abs_n = n.unsigned_abs();
        // for _ in 1..abs_n {
        //     result = result * self;
        // }
        // if n < 0 {
        //     Self::one() / result
        // } else {
        //     result
        // }
    }

    #[inline]
    fn powf(self, n: Self::RealField) -> Self {
        let out_value = self.value().powf(n.value());
        GlobalComputationGraph32::get().add_node(
            NodeType32::Powf,
            out_value,
            Some(self.value()),
            Some(n.value()),
            Some(self.node_idx),
            Some(n.node_idx),
        )
    }

    #[inline]
    fn powc(self, n: Self) -> Self {
        return ComplexField::powf(self, n);
    }

    #[inline]
    fn cbrt(self) -> Self {
        return ComplexField::powf(self, Self::constant(1.0 / 3.0));
    }

    fn is_finite(&self) -> bool {
        return self.value().is_finite();
    }

    fn try_sqrt(self) -> Option<Self> {
        Some(ComplexField::sqrt(self))
    }
}

impl SubsetOf<Self> for adr32 {
    fn to_superset(&self) -> Self {
        self.clone()
    }

    fn from_superset_unchecked(element: &adr32) -> Self {
        element.clone()
    }

    fn is_in_subset(_element: &adr32) -> bool {
        true
    }
}

impl Field for adr32 {}

impl PrimitiveSimdValue for adr32 {}

impl SubsetOf<adr32> for f32 {
    fn to_superset(&self) -> adr32 {
        adr32::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr32) -> Self {
        element.value()
    }

    fn is_in_subset(_: &adr32) -> bool {
        false
    }
}

impl SubsetOf<adr32> for f64 {
    fn to_superset(&self) -> adr32 {
        adr32::constant(*self)
    }

    fn from_superset_unchecked(element: &adr32) -> Self {
        element.value() as f64
    }

    fn is_in_subset(_: &adr32) -> bool {
        false
    }
}

impl SubsetOf<adr32> for u32 {
    fn to_superset(&self) -> adr32 {
        adr32::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr32) -> Self {
        element.value() as u32
    }

    fn is_in_subset(_: &adr32) -> bool {
        false
    }
}

impl SubsetOf<adr32> for u64 {
    fn to_superset(&self) -> adr32 {
        adr32::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr32) -> Self {
        element.value() as u64
    }

    fn is_in_subset(_: &adr32) -> bool {
        false
    }
}

impl SubsetOf<adr32> for u128 {
    fn to_superset(&self) -> adr32 {
        adr32::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr32) -> Self {
        element.value() as u128
    }

    fn is_in_subset(_: &adr32) -> bool {
        false
    }
}

impl SubsetOf<adr32> for i32 {
    fn to_superset(&self) -> adr32 {
        adr32::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr32) -> Self {
        element.value() as i32
    }

    fn is_in_subset(_: &adr32) -> bool {
        false
    }
}

impl SubsetOf<adr32> for i64 {
    fn to_superset(&self) -> adr32 {
        adr32::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr32) -> Self {
        element.value() as i64
    }

    fn is_in_subset(_: &adr32) -> bool {
        false
    }
}

impl SubsetOf<adr32> for i128 {
    fn to_superset(&self) -> adr32 {
        adr32::constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adr32) -> Self {
        element.value() as i128
    }

    fn is_in_subset(_: &adr32) -> bool {
        false
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

unsafe impl Dim for adr32 {
    fn try_to_usize() -> Option<usize> {
        unimplemented!()
    }

    fn value(&self) -> usize {
        unimplemented!()
    }

    fn from_usize(_dim: usize) -> Self {
        unimplemented!()
    }
}
