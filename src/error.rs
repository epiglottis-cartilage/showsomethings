pub type Err = Box<dyn core::error::Error>;
pub type Result<T> = core::result::Result<T, Err>;

// impl core::error::Error for Err {}

#[allow(dead_code)]
pub trait ErrorLogger<T, E: core::fmt::Display> {
    fn log(self) -> core::result::Result<T, E>;
}
impl<T, E: core::fmt::Display> ErrorLogger<T, E> for core::result::Result<T, E> {
    fn log(self) -> core::result::Result<T, E> {
        match self {
            Ok(x) => {
                log::trace!("succ get {}", std::any::type_name::<T>());
                Ok(x)
            }
            Err(e) => {
                log::error!("fail get {}\n{}", std::any::type_name::<T>(), &e);
                #[cfg(debug_assertions)]
                panic!("Aborted");
                #[cfg(not(debug_assertions))]
                Err(e.into())
            }
        }
    }
}
impl<T> ErrorLogger<T, Err> for core::option::Option<T> {
    fn log(self) -> core::result::Result<T, Err> {
        match self {
            Some(x) => {
                log::trace!("succ get {}", std::any::type_name::<T>());
                Ok(x)
            }
            None => {
                log::error!("fail get {}", std::any::type_name::<T>());
                #[cfg(debug_assertions)]
                panic!("None");
                #[allow(unreachable_code)]
                Err(String::from("None").into())
            }
        }
    }
}
