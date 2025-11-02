pub type Result<T, E: core::fmt::Display = Box<dyn core::error::Error>> =
    core::result::Result<T, E>;

pub trait ErrorLogger {
    type Output;
    type Error;
    fn log(self) -> core::result::Result<Self::Output, Self::Error>;
}

impl<T, E: core::fmt::Display> ErrorLogger for core::result::Result<T, E> {
    type Output = T;
    type Error = E;
    fn log(self) -> Self {
        match &self {
            Ok(_) => {
                #[cfg(debug_assertions)]
                log::trace!("succ: {}", std::any::type_name::<T>());
            }
            Err(e) => {
                log::error!("fail {} because: {}", std::any::type_name::<T>(), e);
                #[cfg(debug_assertions)]
                panic!()
            }
        }
        self
    }
}

// 为 Option 实现
impl<T> ErrorLogger for core::option::Option<T> {
    type Output = T;
    type Error = Box<dyn core::error::Error>;
    fn log(self) -> core::result::Result<Self::Output, Self::Error> {
        match &self {
            Some(_) => {
                #[cfg(debug_assertions)]
                log::trace!("succ: {}", std::any::type_name::<T>());
            }
            None => {
                log::error!("fail {} bacuse: None", std::any::type_name::<T>());
                #[cfg(debug_assertions)]
                panic!()
            }
        }
        self.ok_or(String::from("None").into())
    }
}
